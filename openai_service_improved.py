import os
import json
import logging
import re
from typing import List, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIServiceError(Exception):
    """Custom exception for OpenAI service errors"""
    pass

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def parse_srt_time_to_seconds(time_str: str) -> int:
    """Convert SRT timestamp to seconds."""
    try:
        if '.' in time_str:
            time_str = time_str.split('.')[0]
        elif ',' in time_str:
            time_str = time_str.split(',')[0]
        
        time_parts = time_str.split(':')
        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
            return hours * 3600 + minutes * 60 + seconds
        return 0
    except (ValueError, IndexError):
        return 0

def generate_timestamps_for_long_video_simple(srt_entries: List[Dict[str, str]], context: str = None) -> List[str]:
    """
    Generate timestamps for long videos using a simpler approach.
    Sample entries throughout the video and generate comprehensive timestamps.
    """
    if not openai_client:
        raise OpenAIServiceError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
    
    if not srt_entries:
        raise OpenAIServiceError("No SRT entries provided")
    
    total_duration = parse_srt_time_to_seconds(srt_entries[-1]['end']) / 60.0
    logger.info(f"Processing long video: {total_duration:.1f} minutes with {len(srt_entries)} entries")
    
    # Sample entries throughout the video for better coverage
    # Be more conservative with token usage for very long videos
    sample_size = min(1500, len(srt_entries))  # More conservative limit
    step = max(1, len(srt_entries) // sample_size)
    sampled_entries = srt_entries[::step]
    
    # Further reduce if the content is still too large
    total_chars = sum(len(entry['text']) for entry in sampled_entries)
    if total_chars > 30000:  # Much more conservative - roughly 10k tokens
        reduction_factor = 30000 / total_chars
        new_sample_size = int(len(sampled_entries) * reduction_factor)
        step = max(1, len(sampled_entries) // new_sample_size)
        sampled_entries = sampled_entries[::step]
    
    logger.info(f"Final content size: {sum(len(entry['text']) for entry in sampled_entries)} characters")
    
    logger.info(f"Sampling {len(sampled_entries)} entries from {len(srt_entries)} total entries")
    
    # Format for OpenAI
    formatted_lines = []
    for entry in sampled_entries:
        formatted_lines.append(f"[{entry['start']}] {entry['text']}")
    
    formatted_content = '\n'.join(formatted_lines)
    
    # Calculate expected number of timestamps - much more detailed for long content
    # Aim for one timestamp every 3-5 minutes for better granularity
    expected_timestamps = max(16, int(total_duration / 4))  # One every 4 minutes for detailed breakdown
    
    system_prompt = f"""You are an expert at analyzing video content and identifying topic segments from subtitles.

Analyze the provided subtitle content and create {expected_timestamps} topic-based timestamps.

Return a JSON object with a "timestamps" array containing objects with "time" and "description" fields.

Requirements:
- Generate exactly {expected_timestamps} timestamps
- Use timestamps from the subtitle content
- Use format "H:MM:SS" for time
- Keep descriptions under 100 characters
- Spread timestamps across the video duration

Example:
{{
  "timestamps": [
    {{"time": "1:15:30", "description": "Discussion about main topic"}},
    {{"time": "2:45:00", "description": "Transition to new subject"}}
  ]
}}"""

    user_prompt = f"""Analyze this {total_duration:.0f}-minute video content and create {expected_timestamps} timestamps:

{formatted_content}

Context: {context if context else 'General video content'}

Generate {expected_timestamps} timestamps spread across the full duration."""

    try:
        logger.info(f"Sending request to OpenAI for long video analysis")
        
        response = openai_client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        logger.debug(f"Raw OpenAI response: {response}")
        logger.debug(f"Response content: {response_content}")
        
        if not response_content:
            logger.error("Empty response content from OpenAI")
            # Try to get more info about the response
            logger.error(f"Full response object: {response}")
            raise OpenAIServiceError("Empty response from OpenAI")
        
        logger.debug(f"Received response from OpenAI: {response_content[:300]}...")
        
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
            
            # Sort timestamps by time
            def safe_sort_key(timestamp_str):
                try:
                    time_part = timestamp_str.split(' - ')[0]
                    return parse_srt_time_to_seconds(time_part)
                except (IndexError, ValueError):
                    return 0
            
            formatted_timestamps.sort(key=safe_sort_key)
            
            logger.info(f"Successfully generated {len(formatted_timestamps)} topic timestamps for long video")
            return formatted_timestamps
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI JSON response: {str(e)}")
            
            # Fallback: extract any timestamps from the response
            time_pattern = r'\d{1,2}:\d{2}:\d{2}'
            found_times = re.findall(time_pattern, response_content)
            if found_times:
                logger.info(f"Using fallback extraction: found {len(found_times)} timestamps")
                fallback_timestamps = []
                for i, time_stamp in enumerate(found_times[:expected_timestamps]):
                    fallback_timestamps.append(f"{time_stamp} - Topic segment {i+1}")
                return fallback_timestamps
            
            raise OpenAIServiceError(f"Invalid JSON response from OpenAI: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in long video timestamp generation: {str(e)}")
        raise OpenAIServiceError(f"Failed to generate timestamps: {str(e)}")