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
    Generate timestamps for long videos using a chunked approach.
    Processes video in segments to ensure even distribution throughout entire duration.
    """
    if not openai_client:
        raise OpenAIServiceError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
    
    if not srt_entries:
        raise OpenAIServiceError("No SRT entries provided")
    
    total_duration_seconds = parse_srt_time_to_seconds(srt_entries[-1]['end'])
    total_duration_minutes = total_duration_seconds / 60.0
    logger.info(f"Processing long video: {total_duration_minutes:.1f} minutes with {len(srt_entries)} entries")
    
    # Split video into 30-minute chunks for processing
    chunk_duration_minutes = 30
    chunk_duration_seconds = chunk_duration_minutes * 60
    
    all_timestamps = []
    
    # Calculate number of chunks
    num_chunks = max(1, int(total_duration_seconds / chunk_duration_seconds))
    if total_duration_seconds % chunk_duration_seconds > 600:  # If remainder > 10 minutes, add another chunk
        num_chunks += 1
    
    logger.info(f"Processing video in {num_chunks} chunks of {chunk_duration_minutes} minutes each")
    
    for chunk_index in range(num_chunks):
        start_time = chunk_index * chunk_duration_seconds
        end_time = min((chunk_index + 1) * chunk_duration_seconds, total_duration_seconds)
        
        # Get entries for this time chunk
        chunk_entries = []
        for entry in srt_entries:
            entry_start = parse_srt_time_to_seconds(entry['start'])
            if start_time <= entry_start < end_time:
                chunk_entries.append(entry)
        
        if not chunk_entries:
            logger.warning(f"No entries found for chunk {chunk_index + 1}")
            continue
        
        # Sample entries if chunk is too large
        if len(chunk_entries) > 400:
            step = len(chunk_entries) // 400
            chunk_entries = chunk_entries[::step]
        
        logger.info(f"Processing chunk {chunk_index + 1}/{num_chunks}: {len(chunk_entries)} entries ({start_time//60:.0f}-{end_time//60:.0f} minutes)")
        
        # Generate timestamps for this chunk
        chunk_timestamps = process_video_chunk(chunk_entries, chunk_index, num_chunks, context)
        all_timestamps.extend(chunk_timestamps)
    
    # Sort all timestamps by time
    def safe_sort_key(timestamp_str):
        try:
            time_part = timestamp_str.split(' - ')[0]
            return parse_srt_time_to_seconds(time_part)
        except (IndexError, ValueError):
            return 0
    
    all_timestamps.sort(key=safe_sort_key)
    
    logger.info(f"Successfully generated {len(all_timestamps)} timestamps across {num_chunks} chunks")
    return all_timestamps


def process_video_chunk(chunk_entries: List[Dict[str, str]], chunk_index: int, total_chunks: int, context: str = None) -> List[str]:
    """Process a single chunk of video entries to generate timestamps."""
    if not chunk_entries:
        return []
    
    # Format entries for AI
    formatted_lines = []
    for entry in chunk_entries:
        formatted_lines.append(f"[{entry['start']}] {entry['text']}")
    
    formatted_content = '\n'.join(formatted_lines)
    
    system_prompt = f"""You are an expert at analyzing conversation content and identifying when topics change.

This is chunk {chunk_index + 1} of {total_chunks} from a longer podcast/video.

Read through the entire conversation segment and identify EVERY moment where the topic actually changes. Do not limit yourself to a specific number of timestamps.

Return a JSON object with a "timestamps" array containing objects with "time" and "description" fields.

CRITICAL REQUIREMENTS:
- Find ALL genuine topic changes in this segment, not just a few
- A topic change is when the conversation shifts to a completely different subject
- Use exact timestamps that appear in the subtitle content  
- Use format "H:MM:SS" for time (e.g., "1:15:30", "2:45:00")
- Keep descriptions under 100 characters and be specific about the new topic
- Look for these indicators:
  * New subjects being introduced ("So let's talk about...")
  * Clear transitions ("Speaking of...", "That reminds me...", "Moving on...")
  * Shifts from one topic to another (politics to sports, tech to relationships, etc.)
  * New stories, anecdotes, or examples being introduced
  * Questions that introduce new topics

If there are 20 topic changes, return 20 timestamps. If there are 3, return 3. Be comprehensive.

Example:
{{
  "timestamps": [
    {{"time": "1:15:30", "description": "Shift from AI discussion to dating app experiences"}},
    {{"time": "1:28:45", "description": "New topic: Universal basic income debate"}},
    {{"time": "1:42:15", "description": "Story about weird restaurant experience"}},
    {{"time": "1:55:30", "description": "Discussion of new movie releases"}}
  ]
}}"""

    user_prompt = f"""Analyze this conversation segment and identify ALL genuine topic changes:

{formatted_content}

Context: {context if context else 'Podcast/video content'}

Find every moment where the conversation shifts to a new topic - could be 1 timestamp, could be 20 timestamps, depending on how many topic changes actually occur in this segment."""

    try:
        logger.info(f"Sending request to OpenAI for chunk {chunk_index + 1}")
        logger.debug(f"Chunk {chunk_index + 1} content length: {len(formatted_content)} characters")
        
        if not openai_client:
            logger.error("OpenAI client is None - API key not configured")
            return []
        
        response = openai_client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        logger.debug(f"Raw response for chunk {chunk_index + 1}: {response}")
        
        response_content = response.choices[0].message.content
        logger.debug(f"Response content for chunk {chunk_index + 1}: {response_content}")
        
        if not response_content:
            logger.error(f"Empty response content for chunk {chunk_index + 1}")
            logger.error(f"Full response object: {response}")
            return []
        
        try:
            result = json.loads(response_content)
            timestamps_data = result.get("timestamps", [])
            
            logger.info(f"Parsed {len(timestamps_data)} timestamps from chunk {chunk_index + 1}")
            
            formatted_timestamps = []
            for item in timestamps_data:
                time = item.get("time", "")
                description = item.get("description", "")
                
                if time and description:
                    formatted_timestamps.append(f"{time} - {description}")
                else:
                    logger.warning(f"Skipping invalid timestamp in chunk {chunk_index + 1}: {item}")
            
            logger.info(f"Generated {len(formatted_timestamps)} valid timestamps for chunk {chunk_index + 1}")
            return formatted_timestamps
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for chunk {chunk_index + 1}: {str(e)}")
            logger.error(f"Raw content that failed to parse: {response_content}")
            return []
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index + 1}: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []