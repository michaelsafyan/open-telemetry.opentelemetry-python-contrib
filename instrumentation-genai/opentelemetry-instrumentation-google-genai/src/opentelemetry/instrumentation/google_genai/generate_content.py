from typing import Union, Optional, Iterator, AsyncIterator, Awaitable

import logging
import time
import os
import functools


from google.genai.models import Models, AsyncModels
from google.genai.types import (
    ContentListUnion,
    ContentListUnionDict,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
)
from .flags import is_content_recording_enabled
from .otel_wrapper import OTelWrapper

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes import error_attributes


_logger = logging.getLogger(__name__)


# Enable these after these cases are fully vetted and tested
_INSTRUMENT_STREAMING = False
_INSTRUMENT_ASYNC = False



class _MethodsSnapshot:

    def __init__(self):
        self._original_generate_content = Models.generate_content
        self._original_generate_content_stream = Models.generate_content_stream
        self._original_async_generate_content = AsyncModels.generate_content
        self._original_async_generate_content_stream = AsyncModels.generate_content_stream

    @property
    def generate_content(self):
        return self._original_generate_content

    @property
    def generate_content_stream(self):
        return self._original_generate_content_stream

    @property
    def async_generate_content(self):
        return self._original_async_generate_content

    @property
    def async_generate_content_stream(self):
        return self._original_async_generate_content_stream

    def restore(self):
        Models.generate_content = self._original_generate_content
        Models.generate_content_stream = self._original_generate_content_stream
        AsyncModels.generate_content = self._original_async_generate_content
        AsyncModels.generate_content_stream = self._original_async_generate_content_stream


def _get_vertexai_system_name():
    return gen_ai_attributes.GenAiSystemValues.VERTEX_AI.name.lower()


def _get_gemini_system_name():
    return gen_ai_attributes.GenAiSystemValues.GEMINI.name.lower()


def _guess_genai_system_from_env():
    if os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '0').lower() in [
          'true',
          '1',
      ]:
      return _get_vertexai_system_name()
    return _get_gemini_system_name()



def _determine_genai_system(models_object: Union[Models, AsyncModels]):
    client = getattr(models_object, '_api_client')
    if not client:
        return _guess_genai_system_from_env()
    vertexai_attr = getattr(client, 'vertexai')
    if vertexai_attr is None:
        return _guess_genai_system_from_env()
    if vertexai_attr:
        return _get_vertexai_system_name()
    return _get_gemini_system_name()



def _get_config_property(
    config: Optional[GenerateContentConfigOrDict],
    path: str):
    if config is None:
        return None
    path_segments = path.split('.')
    current_context = config
    for path_segment in path_segments:
        if current_context is None:
            return None
        if isinstance(current_context, dict):
            current_context = current_context.get(path_segment)
        else:
            current_context = getattr(current_context, path_segment)
    return current_context


def _get_response_property(
    response: GenerateContentResponse,
    path: str):
    path_segments = path.split('.')
    current_context = response
    for path_segment in path_segments:
        if current_context is None:
            return None
        if isinstance(current_context, dict):
            current_context = current_context.get(path_segment)
        else:
            current_context = getattr(current_context, path_segment)
    return current_context


def _get_temperature(config: Optional[GenerateContentConfigOrDict]):
    return _get_config_property(config, 'temperature')


def _get_top_k(config: Optional[GenerateContentConfigOrDict]):
    return _get_config_property(config, 'top_k')


def _get_top_p(config: Optional[GenerateContentConfigOrDict]):
    return _get_config_property(config, 'top_p')


_SPAN_ATTRIBUTE_TO_CONFIG_EXTRACTOR = {
    gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE: _get_temperature,
    gen_ai_attributes.GEN_AI_REQUEST_TOP_K: _get_top_k,
    gen_ai_attributes.GEN_AI_REQUEST_TOP_P: _get_top_p,
}


class _GenerateContentInstrumentationHelper:

    def __init__(
        self,
        models_object: Union[Models, AsyncModels],
        otel_wrapper: OTelWrapper,
        model: str):
        self._start_time = time.time_ns()
        self._otel_wrapper = otel_wrapper
        self._genai_system = _determine_genai_system(models_object)
        self._genai_request_model = model
        self._finish_reasons_set = set()
        self._error_type = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._content_recording_enabled = is_content_recording_enabled()

    def start_span_as_current_span(self, name):
        return self._otel_wrapper.start_as_current_span(
            name,
            start_time=self._start_time,
            attributes={
                gen_ai_attributes.GEN_AI_SYSTEM: self._genai_system,
                gen_ai_attributes.GEN_AI_REQUEST_MODEL: self._genai_request_model,
                gen_ai_attributes.GEN_AI_OPERATION_NAME: 'GenerateContent',
            }
        )

    def process_request(
        self,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict]):
        span = trace.get_current_span()
        for attribute_key, extractor in _SPAN_ATTRIBUTE_TO_CONFIG_EXTRACTOR.items():
            attribute_value = extractor(config)
            if attribute_value is not None:
                span.set_attribute(attribute_key, attribute_value)
        self._maybe_log_system_instruction(config=config)
        self._maybe_log_user_prompt(contents)
        

    def process_response(self, response: GenerateContentResponse):
        self._maybe_update_token_counts(response)
        self._maybe_update_error_type(response)
        self._maybe_log_response(response)

    def process_error(self, e: Exception):
        self._error_type = str(e.__class__.__name__)

    def finalize_processing(self):
        span = trace.get_current_span()
        span.set_attribute(gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, self._input_tokens)
        span.set_attribute(gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS, self._output_tokens)
        span.set_attribute(gen_ai_attributes.GEN_AI_RESPONSE_FINISH_REASONS, sorted(list(self._finish_reasons_set)))
        self._record_token_usage_metric()
        self._record_duration_metric()
        self._otel_wrapper.done()

    def _maybe_update_token_counts(self, response: GenerateContentResponse):
        input_tokens = _get_response_property(response, 'usage_metadata.prompt_token_count')
        output_tokens = _get_response_property(response, 'usage_metadata.candidates_token_count')
        if input_tokens:
            self._input_tokens += input_tokens
        if output_tokens:
            self._output_tokens += output_tokens

    def _maybe_update_error_type(self, response: GenerateContentResponse):
        if response.candidates:
            return
        if ((not response.prompt_feedback) or
            (not response.prompt_feedback.block_reason) or
            (block_reason == genai_types.BlockedReason.BLOCKED_REASON_UNSPECIFIED)):
            self._error_type = 'NO_CANDIDATES'
            return
        block_reason = response.prompt_feedback.block_reason
        self._error_type = 'BLOCKED_{}'.format(block_reason.name)

    def _maybe_log_system_instruction(self, config: Optional[GenerateContentConfigOrDict]=None):
        if not self._content_recording_enabled:
            return
        system_instruction = _get_config_property(config, 'system_instruction')
        if not system_instruction:
            return
        self._otel_wrapper.log_system_prompt(
            attributes={
                gen_ai_attributes.GEN_AI_SYSTEM: self._genai_system,
            },
            body={
                'content': system_instruction,
            })
    
    def _maybe_log_user_prompt(self, contents: Union[ContentListUnion, ContentListUnionDict]):
        if not self._content_recording_enabled:
            return
        self._otel_wrapper.log_user_prompt(
            attributes={
                gen_ai_attributes.GEN_AI_SYSTEM: self._genai_system,
            },
            body={
                'content': contents,
            })

    def _maybe_log_response(self, response: GenerateContentResponse):
        if not self._content_recording_enabled:
            return
        self._otel_wrapper.log_response_content(
            attributes={
                gen_ai_attributes.GEN_AI_SYSTEM: self._genai_system,
            },
            body={
                'content': response.model_dump(),
            })

    def _record_token_usage_metric(self):
        self._otel_wrapper.token_usage_metric.record(
            self._input_tokens,
            attributes={
                gen_ai_attributes.GEN_AI_TOKEN_TYPE: 'input',
                gen_ai_attributes.GEN_AI_SYSTEM: self._genai_system,
                gen_ai_attributes.GEN_AI_REQUEST_MODEL: self._genai_request_model,
                gen_ai_attributes.GEN_AI_OPERATION_NAME: 'GenerateContent',
            }
        )
        self._otel_wrapper.token_usage_metric.record(
            self._output_tokens,
            attributes={
                gen_ai_attributes.GEN_AI_TOKEN_TYPE: 'output',
                gen_ai_attributes.GEN_AI_SYSTEM: self._genai_system,
                gen_ai_attributes.GEN_AI_REQUEST_MODEL: self._genai_request_model,
                gen_ai_attributes.GEN_AI_OPERATION_NAME: 'GenerateContent',
            }
        )
    
    def _record_duration_metric(self):
        attributes = {
            gen_ai_attributes.GEN_AI_SYSTEM: self._genai_system,
            gen_ai_attributes.GEN_AI_REQUEST_MODEL: self._genai_request_model,
            gen_ai_attributes.GEN_AI_OPERATION_NAME: 'GenerateContent',
        }
        if self._error_type is not None:
            attributes[error_attributes.ERROR_TYPE] = self._error_type
        duration_nanos = time.time_ns() - self._start_time
        duration_seconds = duration_nanos / 1e9
        self._otel_wrapper.operation_duration_metric.record(
            duration_seconds,
            attributes=attributes,
        )



def _create_instrumented_generate_content(
    snapshot: _MethodsSnapshot,
    otel_wrapper: OTelWrapper):
    wrapped_func = snapshot.generate_content
    @functools.wraps(wrapped_func)
    def instrumented_generate_content(
        self: Models,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None) -> GenerateContentResponse:
        helper = _GenerateContentInstrumentationHelper(self, otel_wrapper, model)
        with helper.start_span_as_current_span('google.genai.Models.generate_content'):
            helper.process_request(contents, config)
            try:
                response = wrapped_func(self, model=model, contents=contents, config=config)
                helper.process_response(response)
                return response
            except Exception as e:
                helper.process_error(e)
                raise
            finally:
                helper.finalize_processing()
    return instrumented_generate_content


def _create_instrumented_generate_content_stream(
    snapshot: _MethodsSnapshot,
    otel_wrapper: OTelWrapper):
    wrapped_func = snapshot.generate_content_stream
    if not _INSTRUMENT_STREAMING:
        # TODO: remove once this case has been fully tested
        return wrapped_func
    @functools.wraps(wrapped_func)
    def instrumented_generate_content_stream(
        self: Models,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None) -> Iterator[GenerateContentResponse]:
        helper = _GenerateContentInstrumentationHelper(self, otel_wrapper, model)
        with helper.start_span_as_current_span('google.genai.Models.generate_content_stream'):
            helper.process_request(contents, config)
            try:
                for response in wrapped_func(self, model=model, contents=contents, config=config):
                    helper.process_response(response)
                    yield response
            except Exception as e:
                helper.process_error(e)
                raise
            finally:
                helper.finalize_processing()
    return instrumented_generate_content_stream


def _create_instrumented_async_generate_content(
    snapshot: _MethodsSnapshot,
    otel_wrapper: OTelWrapper):
    wrapped_func = snapshot.async_generate_content
    if not _INSTRUMENT_ASYNC:
        # TODO: remove once this case has been fully tested
        return wrapped_func
    @functools.wraps(wrapped_func)
    async def instrumented_generate_content(
        self: AsyncModels,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None) -> GenerateContentResponse:
        helper = _GenerateContentInstrumentationHelper(self, otel_wrapper, model)
        with helper.start_span_as_current_span('google.genai.AsyncModels.generate_content'):
            helper.process_request(contents, config)
            try:
                response = await wrapped_func(self, model=model, contents=contents, config=config)
                helper.process_response(response)
                return response
            except Exception as e:
                helper.process_error(e)
                raise
            finally:
                helper.finalize_processing()
    return instrumented_generate_content


def _create_instrumented_async_generate_content_stream(
    snapshot: _MethodsSnapshot,
    otel_wrapper: OTelWrapper):
    wrapped_func = snapshot.async_generate_content_stream
    if not _INSTRUMENT_ASYNC or not _INSTRUMENT_STREAMING:
        # TODO: remove once this case has been fully tested
        return wrapped_func
    @functools.wraps(wrapped_func)
    async def instrumented_generate_content_stream(
        self: AsyncModels,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None) -> Awaitable[AsyncIterator[GenerateContentResponse]]:
        helper = _GenerateContentInstrumentationHelper(self, otel_wrapper, model)
        with helper.start_span_as_current_span('google.genai.AsyncModels.generate_content_stream'):
            helper.process_request(contents, config)
            try:
                async for response in wrapped_func(self, model=model, contents=contents, config=config):
                    helper.process_response(response)
                    yield response
            except Exception as e:
                helper.process_error(e)
                raise
            finally:
                helper.finalize_processing()
    return instrumented_generate_content_stream


def uninstrument_generate_content(snapshot: object):
    assert isinstance(snapshot, _MethodsSnapshot)
    snapshot.restore()


def instrument_generate_content(otel_wrapper: OTelWrapper) -> object:
    snapshot = _MethodsSnapshot()
    Models.generate_content = _create_instrumented_generate_content(snapshot, otel_wrapper)
    Models.generate_content_stream = _create_instrumented_generate_content_stream(snapshot, otel_wrapper)
    AsyncModels.generate_content = _create_instrumented_async_generate_content(snapshot, otel_wrapper)
    AsyncModels.generate_content_stream = _create_instrumented_async_generate_content_stream(snapshot, otel_wrapper)
    return snapshot