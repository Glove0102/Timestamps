import os
import json
import logging
from typing import List, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIServiceError(Exception):
    """Custom exception for OpenAI service errors"""
    pass

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def generate_topic_timestamps(srt_entries: List[Dict[str, str]], context: str = None) -> List[str]:
    """
    Generate topic-based timestamps using OpenAI API.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries with start, end, text
        context (str, optional): Additional context for better topic detection
        
    Returns:
        List[str]: List of formatted timestamps with topic descriptions
        
    Raises:
        OpenAIServiceError: If OpenAI API call fails or returns invalid response
    """
    if not openai_client:
        raise OpenAIServiceError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
    
    try:
        # Format SRT content for analysis
        formatted_content = format_srt_for_openai(srt_entries)
        
        # Build the system prompt
        system_prompt = """You are an expert at analyzing video content and identifying topic segments from subtitles. 
Your task is to analyze subtitle text with timestamps and identify distinct topic segments or content changes.

Return a JSON object with a "timestamps" array containing objects with "time" and "description" fields.
Each timestamp should mark the beginning of a new topic or significant content shift.

Guidelines:
- Cover the entire content duration with timestamps. IF the content ends before the last timestamp, include a final timestamp at the end.
- Use the format "H:MM:SS" for timestamps (e.g., "0:00:15", "0:18:11", "1:23:45")
- Descriptions should be brief but descriptive (20-60 characters)
- Focus on meaningful content transitions, not minor topic shifts
- Include 3-15 topic segments depending on content length
- Start with "0:00:00" if the content begins immediately

Example output format:
{
  "timestamps": [
    {"time": "0:00:00", "description": "Introduction and sponsor mentions"},
    {"time": "0:02:15", "description": "Main topic discussion begins"},
    {"time": "0:15:30", "description": "Technical details and examples"},
    {"time": "0:28:45", "description": "Q&A and closing remarks"}
  ]
}"""

        # Build the user prompt
        user_prompt = f"Analyze the following subtitle content and identify topic segments:\n\n{formatted_content}"
        
        if context:
            user_prompt += f"\n\nAdditional context: {context}"
        
        user_prompt += "\n\nGenerate topic-based timestamps in JSON format as specified."
        
        logger.debug(f"Sending request to OpenAI with {len(srt_entries)} subtitle entries")
        
        # Make API call to OpenAI
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=5000,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        response_content = response.choices[0].message.content
        logger.debug(f"Received response from OpenAI: {response_content[:200]}...")
        
        try:
            result = json.loads(response_content)
            timestamps_data = result.get("timestamps", [])
            
            if not timestamps_data:
                raise OpenAIServiceError("No timestamps found in OpenAI response")
            
            # Format timestamps for display
            formatted_timestamps = []
            for item in timestamps_data:
                time = item.get("time", "")
                description = item.get("description", "")
                
                if time and description:
                    formatted_timestamps.append(f"{time} - {description}")
                else:
                    logger.warning(f"Skipping malformed timestamp entry: {item}")
            
            if not formatted_timestamps:
                raise OpenAIServiceError("No valid timestamps generated")
            
            logger.info(f"Successfully generated {len(formatted_timestamps)} topic timestamps")
            return formatted_timestamps
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI JSON response: {str(e)}")
            raise OpenAIServiceError(f"Invalid JSON response from OpenAI: {str(e)}")
    
    except OpenAIServiceError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI service: {str(e)}")
        raise OpenAIServiceError(f"Failed to generate timestamps: {str(e)}")

def format_srt_for_openai(srt_entries: List[Dict[str, str]]) -> str:
    """
    Format SRT entries for OpenAI analysis.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries
        
    Returns:
        str: Formatted content for OpenAI analysis
    """
    formatted_lines = []
    
    for entry in srt_entries:
        # Include timestamp and text for analysis
        formatted_lines.append(f"[{entry['start']}] {entry['text']}")
    
    return '\n'.join(formatted_lines)
