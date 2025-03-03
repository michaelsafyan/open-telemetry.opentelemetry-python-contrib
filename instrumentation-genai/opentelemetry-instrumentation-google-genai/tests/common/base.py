# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import os
import unittest
import unittest.mock

import google.genai
import google.genai.types as genai_types
from google.genai.models import Models, AsyncModels

from .instrumentation_context import InstrumentationContext
from .otel_mocker import OTelMocker
from .requests_mocker import RequestsMocker


class _FakeCredentials(google.auth.credentials.AnonymousCredentials):
    def refresh(self, request):
        pass


class TestCase(unittest.TestCase):

    def setUp(self):
        self._otel = OTelMocker()
        self._otel.install()
        self._requests = RequestsMocker()
        self._requests.install()
        self._instrumentation_context = None
        self._api_key = "test-api-key"
        self._project = "test-project"
        self._location = "test-location"
        self._client = None
        self._uses_vertex = False
        self._credentials = _FakeCredentials()
        self._generate_content_mock = None
        self._generate_content_stream_mock = None
        self._original_generate_content = Models.generate_content
        self._original_generate_content_stream = Models.generate_content_stream
        self._original_async_generate_content = AsyncModels.generate_content
        self._original_async_generate_content_stream = (
            AsyncModels.generate_content_stream
        )

    def _lazy_init(self):
        self._instrumentation_context = InstrumentationContext()
        self._instrumentation_context.install()

    @property
    def mock_generate_content(self):
        if self._generate_content_mock is None:
            self._create_mocks()
        return self._generate_content_mock
    
    @property
    def mock_generate_content_stream(self):
        if self._generate_content_stream_mock is None:
            self._create_mocks()
        return self._generate_content_stream_mock

    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def requests(self):
        return self._requests

    @property
    def otel(self):
        return self._otel

    def set_use_vertex(self, use_vertex):
        self._uses_vertex = use_vertex

    def generate_content_response(
        self,
        part: Optional[genai_types.Part] = None,
        parts: Optional[list[genai_types.Part]] = None,
        content: Optional[genai_types.Content] = None,
        candidate: Optional[genai_types.Candidate] = None,
        candidates: Optional[list[genai_types.Candidate]] = None,
        text: Optional[str] = None):
        if text is None:
            text = 'Some response text'
        if part is None:
            part = genai_types.Part(text=text)
        if parts is None:
            parts = [part]
        if content is None:
            content = genai_types.Content(parts=parts, role='model')
        if candidate is None:
            candidate = genai_types.Candidate(content=content)
        if candidates is None:
            candidates = [candidate]
        return genai_types.GenerateContentResponse(candidates=candidates)

    def _create_mocks(self):
        print("Initializing mocks.")
        if self._client is not None:
            self._client = None
        if self._instrumentation_context is not None:
            self._instrumentation_context.uninstall()
            self._instrumentation_context = None
        self._generate_content_mock = unittest.mock.MagicMock()
        self._generate_content_stream_mock = unittest.mock.MagicMock()

        def convert_response(arg):
            if isinstance(arg, genai_types.GenerateContentResponse):
                return arg
            if isinstance(arg, str):
                return self.generate_content_response(text=arg)
            if isinstance(arg, dict):
                try:
                    return genai_types.GenerateContentResponse(**arg)
                except Exception:
                    return self.generate_content_response(**arg)
            return arg
        
        def default_stream(*args, **kwargs):
            result = self._generate_content_mock(*args, **kwargs)
            yield result
        self._generate_content_stream_mock.side_effect = default_stream

        def sync_variant(*args, **kwargs):
            return convert_response(self._generate_content_mock(*args, **kwargs))
        
        def sync_stream_variant(*args, **kwargs):
            print("Calling sync stream variant")
            for result in self._generate_content_stream_mock(*args, **kwargs):
                yield convert_response(result)

        async def async_variant(*args, **kwargs):
            print("Calling async non-streaming variant")
            return sync_variant(*args, **kwargs)

        async def async_stream_variant(*args, **kwargs):
            print("Calling async stream variant")
            async def gen():
                for result in sync_stream_variant(*args, **kwargs):
                    yield result
            class GeneratorProvider:
                def __aiter__(self):
                    return gen()
            return GeneratorProvider()
        Models.generate_content = sync_variant
        Models.generate_content_stream = sync_stream_variant
        AsyncModels.generate_content = async_variant
        AsyncModels.generate_content_stream = async_stream_variant

    def _create_client(self):
        self._lazy_init()
        if self._uses_vertex:
            os.environ["GOOGLE_API_KEY"] = self._api_key
            return google.genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
                credentials=self._credentials,
            )
        return google.genai.Client(api_key=self._api_key)

    def tearDown(self):
        if self._instrumentation_context is not None:
            self._instrumentation_context.uninstall()
        if self._generate_content_mock is None:
            assert Models.generate_content == self._original_generate_content
            assert Models.generate_content_stream == self._original_generate_content_stream
            assert AsyncModels.generate_content == self._original_async_generate_content
            assert AsyncModels.generate_content_stream == self._original_async_generate_content_stream
        self._requests.uninstall()
        self._otel.uninstall()
        Models.generate_content = self._original_generate_content
        Models.generate_content_stream = self._original_generate_content_stream
        AsyncModels.generate_content = self._original_async_generate_content
        AsyncModels.generate_content_stream = (
            self._original_async_generate_content_stream
        )
