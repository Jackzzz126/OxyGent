"""
Integration tests for create_flow_image.py module.

Before using integration tests, make sure to check the following steps:
    + The OpenAI compatible API endpoint is accessible and configured.
    + The API keys and other required environment variables are set.
    + The models are available and accessible.
    + The file system has write permissions for test output.
    + Network connectivity is available for API calls.
Even if we pass the above checks, we still do not guarantee that the synthetic tests will pass 100% of the time for the correct code:
for some specific models, it is really difficult to predict the output of those models, so we only set up some very simple fuzzy matches.
For developers who need to check the quality of their code, we strongly recommend running unit tests first,
and if you have better suggestions for testing, you are welcome to make a pull request.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import requests

from oxygent.chart.create_flow_image import (
    generate_flow_chart,
    call_openai_api,
    extract_mermaid_code,
    generate_sample_mermaid,
    create_html_with_mermaid
)


class TestExtractMermaidCode:
    """Test the extract_mermaid_code function."""
    
    def test_extract_mermaid_with_code_blocks(self):
        """Test extracting mermaid code from properly formatted code blocks."""
        content = """Here is a mermaid diagram:
```mermaid
flowchart TD
    A[Start] --> B[End]
```
That's it."""
        
        result = extract_mermaid_code(content)
        expected = "flowchart TD\n    A[Start] --> B[End]"
        assert result == expected
    
    def test_extract_mermaid_without_code_blocks(self):
        """Test extracting mermaid code without explicit code blocks."""
        content = """flowchart TD
    A[Start] --> B[Process]
    B --> C[End]
    
Some other text here."""
        
        result = extract_mermaid_code(content)
        assert "flowchart TD" in result
        assert "A[Start] --> B[Process]" in result
    
    def test_extract_mermaid_no_valid_content(self):
        """Test extracting mermaid code when no valid content exists."""
        content = "This is just regular text with no mermaid content."
        
        result = extract_mermaid_code(content)
        assert result is None
    
    def test_extract_mermaid_graph_syntax(self):
        """Test extracting mermaid code with graph syntax."""
        content = """graph LR
    A --> B
    B --> C"""
        
        result = extract_mermaid_code(content)
        assert "graph LR" in result
        assert "A --> B" in result


class TestGenerateSampleMermaid:
    """Test the generate_sample_mermaid function."""
    
    def test_generate_sample_mermaid_returns_valid_code(self):
        """Test that generate_sample_mermaid returns valid mermaid code."""
        result = generate_sample_mermaid()
        
        assert isinstance(result, str)
        assert "flowchart TD" in result
        assert "[需求分析]" in result
        assert "[项目完成]" in result
        assert "style" in result
    
    def test_generate_sample_mermaid_extractable(self):
        """Test that the sample mermaid code can be extracted properly."""
        sample_code = generate_sample_mermaid()
        extracted = extract_mermaid_code(sample_code)
        
        assert extracted is not None
        assert "flowchart TD" in extracted


class TestCreateHtmlWithMermaid:
    """Test the create_html_with_mermaid function."""
    
    def test_create_html_with_valid_mermaid(self):
        """Test creating HTML file with valid mermaid code."""
        mermaid_code = "flowchart TD\n    A[Start] --> B[End]"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = create_html_with_mermaid(mermaid_code, tmp_path)
            
            assert result is True
            assert os.path.exists(tmp_path)
            
            # Check file content
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "<!DOCTYPE html>" in content
            assert "Mermaid 流程图编辑器" in content
            assert mermaid_code in content
            assert "mermaid@10" in content
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_create_html_with_invalid_path(self):
        """Test creating HTML file with invalid path."""
        mermaid_code = "flowchart TD\n    A[Start] --> B[End]"
        invalid_path = "/nonexistent/directory/test.html"
        
        result = create_html_with_mermaid(mermaid_code, invalid_path)
        assert result is False


class TestCallOpenaiApi:
    """Test the call_openai_api function."""
    
    @patch('oxygent.chart.create_flow_image.requests.post')
    def test_call_openai_api_success(self, mock_post):
        """Test successful API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "```mermaid\nflowchart TD\n    A[Test] --> B[Success]\n```"
                }
            }]
        }
        mock_post.return_value = mock_response
        
        result = call_openai_api("Generate a test flowchart")
        
        assert "flowchart TD" in result
        assert "A[Test] --> B[Success]" in result
        mock_post.assert_called_once()
    
    @patch('oxygent.chart.create_flow_image.requests.post')
    def test_call_openai_api_failure(self, mock_post):
        """Test API call failure fallback to sample."""
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        
        result = call_openai_api("Generate a test flowchart")
        
        # Should fallback to sample mermaid
        assert "flowchart TD" in result
        assert "[需求分析]" in result  # From sample mermaid
    
    @patch('oxygent.chart.create_flow_image.requests.post')
    def test_call_openai_api_invalid_response(self, mock_post):
        """Test API call with invalid response format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "format"}
        mock_post.return_value = mock_response
        
        result = call_openai_api("Generate a test flowchart")
        
        # Should fallback to sample mermaid
        assert "flowchart TD" in result
        assert "[需求分析]" in result  # From sample mermaid
    
    @patch('oxygent.chart.create_flow_image.requests.post')
    def test_call_openai_api_http_error(self, mock_post):
        """Test API call with HTTP error status."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        result = call_openai_api("Generate a test flowchart")
        
        # Should fallback to sample mermaid
        assert "flowchart TD" in result
        assert "[需求分析]" in result  # From sample mermaid


class TestGenerateFlowChart:
    """Test the generate_flow_chart function."""
    
    @pytest.mark.asyncio
    @patch('oxygent.chart.create_flow_image.call_openai_api')
    @patch('oxygent.chart.create_flow_image.create_html_with_mermaid')
    @patch('oxygent.chart.create_flow_image.webbrowser.open')
    async def test_generate_flow_chart_success(self, mock_browser, mock_create_html, mock_api_call):
        """Test successful flow chart generation."""
        # Mock API response
        mock_api_call.return_value = "flowchart TD\n    A[Start] --> B[End]"
        mock_create_html.return_value = True
        
        description = "Generate a simple flowchart"
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test_flowchart.html")
            
            result = await generate_flow_chart(description, output_path)
            
            assert "✅" in result
            assert "流程图已生成" in result
            assert output_path in result
            
            mock_api_call.assert_called_once_with(description)
            mock_create_html.assert_called_once()
            mock_browser.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('oxygent.chart.create_flow_image.call_openai_api')
    @patch('oxygent.chart.create_flow_image.create_html_with_mermaid')
    async def test_generate_flow_chart_html_creation_failure(self, mock_create_html, mock_api_call):
        """Test flow chart generation when HTML creation fails."""
        mock_api_call.return_value = "flowchart TD\n    A[Start] --> B[End]"
        mock_create_html.return_value = False
        
        description = "Generate a simple flowchart"
        
        result = await generate_flow_chart(description)
        
        assert "❌" in result
        assert "生成流程图时出错" in result
    
    @pytest.mark.asyncio
    @patch('oxygent.chart.create_flow_image.call_openai_api')
    async def test_generate_flow_chart_default_output_path(self, mock_api_call):
        """Test flow chart generation with default output path."""
        mock_api_call.return_value = "flowchart TD\n    A[Start] --> B[End]"
        
        description = "Generate a simple flowchart"
        
        # Mock the file creation to avoid actual file system operations
        with patch('oxygent.chart.create_flow_image.create_html_with_mermaid', return_value=True), \
             patch('oxygent.chart.create_flow_image.webbrowser.open'):
            
            result = await generate_flow_chart(description)
            
            assert "✅" in result or "❌" in result  # Should return some result
            mock_api_call.assert_called_once_with(description)
    
    @pytest.mark.asyncio
    async def test_generate_flow_chart_exception_handling(self):
        """Test flow chart generation exception handling."""
        description = "Generate a simple flowchart"
        
        # Force an exception by providing invalid parameters
        with patch('oxygent.chart.create_flow_image.call_openai_api', side_effect=Exception("Test error")):
            
            result = await generate_flow_chart(description)
            
            assert "❌" in result
            assert "生成流程图时出错" in result


@pytest.mark.asyncio
async def test_generate_flow_chart_integration():
    """Integration test for the complete flow chart generation process."""
    description = "Generate a software development workflow with analysis, design, coding, testing, and deployment stages"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "integration_test_flowchart.html")
        
        # Mock the browser opening to avoid actually opening browser during tests
        with patch('oxygent.chart.create_flow_image.webbrowser.open'):
            result = await generate_flow_chart(description, output_path)
        
        # Should return a result string
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Check if file was created (when not mocked)
        if "✅" in result:
            assert os.path.exists(output_path)
            
            # Verify HTML content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "<!DOCTYPE html>" in content
            assert "Mermaid" in content
            assert "flowchart" in content.lower()


class TestModuleConfiguration:
    """Test module-level configuration and constants."""
    
    def test_api_configuration(self):
        """Test that API configuration constants are properly set."""
        from oxygent.chart.create_flow_image import API_BASE_URL, API_KEY, MODEL_NAME
        
        assert isinstance(API_BASE_URL, str)
        assert len(API_BASE_URL) > 0
        assert isinstance(API_KEY, str)
        assert isinstance(MODEL_NAME, str)
        assert len(MODEL_NAME) > 0
    
    def test_prompt_template(self):
        """Test that the prompt template is properly configured."""
        from oxygent.chart.create_flow_image import DEFAULT_PROMPT_TEMPLATE
        
        assert isinstance(DEFAULT_PROMPT_TEMPLATE, str)
        assert "{description}" in DEFAULT_PROMPT_TEMPLATE
        assert "mermaid" in DEFAULT_PROMPT_TEMPLATE.lower()
        assert "```mermaid" in DEFAULT_PROMPT_TEMPLATE
    
    def test_function_hub_initialization(self):
        """Test that FunctionHub is properly initialized."""
        from oxygent.chart.create_flow_image import flow_image_gen_tools
        
        assert flow_image_gen_tools is not None
        assert hasattr(flow_image_gen_tools, 'name')
        assert flow_image_gen_tools.name == "flow_image_gen_tools"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])