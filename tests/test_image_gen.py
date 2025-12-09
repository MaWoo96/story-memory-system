import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from services.image_gen import ImageGenService

@pytest.fixture
def image_gen_service():
    return ImageGenService()

@pytest.fixture
def mock_entity():
    return {
        &quot;name&quot;: &quot;Mira&quot;,
        &quot;canonical_name&quot;: &quot;mira&quot;,
        &quot;description&quot;: &quot;voluptuous 34yo auburn-haired MILF&quot;,
        &quot;facts&quot;: [&quot;green dress soaked&quot;, &quot;in bakery alley&quot;]
    }

@pytest.fixture
def story_id():
    return &quot;test-story-123&quot;

def test_generate_entity_image_success(image_gen_service, mock_entity, story_id, tmp_path):
    # Patch requests.post
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        &quot;images&quot;: [b&quot;iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==&quot;]  # tiny png base64
    }
    
    with patch(&quot;services.image_gen.requests.post&quot;, return_value=mock_response):
        image_path = image_gen_service.generate_entity_image(mock_entity, story_id)
        
        assert image_path.endswith(&quot;mira.png&quot;)
        assert os.path.exists(image_path)
        assert os.path.getsize(image_path) &gt; 0

def test_generate_entity_image_failure(image_gen_service, mock_entity, story_id):
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception(&quot;API error&quot;)
    
    with patch(&quot;services.image_gen.requests.post&quot;, return_value=mock_response):
        image_path = image_gen_service.generate_entity_image(mock_entity, story_id)
        
        assert &quot;placeholder&quot; in image_path