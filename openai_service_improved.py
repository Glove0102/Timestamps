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

Analyze the conversation and identify ONLY the moments where the topic actually changes - don't generate timestamps based on time intervals.

Return a JSON object with a "timestamps" array containing objects with "time" and "description" fields.

Requirements:
- Only create timestamps when there is a genuine topic change or subject shift
- Use exact timestamps that appear in the subtitle content
- Use format "H:MM:SS" for time (e.g., "1:15:30", "2:45:00")
- Keep descriptions under 100 characters and be specific about what the new topic is
- Look for these indicators of topic changes:
  * New subjects being introduced
  * Transitions like "speaking of...", "that reminds me...", "changing topics..."
  * Clear shifts in conversation focus
  * Introductions of new stories or examples
  * Breaks in conversation flow

Do NOT create timestamps just to fill time - only when topics genuinely change.

Example:
{{
  "timestamps": [
    {{"time": "1:15:30", "description": "Transition from AI discussion to talk about dating apps"}},
    {{"time": "1:28:45", "description": "New topic: Universal basic income debate begins"}}
  ]
}}"""

    user_prompt = f"""Analyze this conversation segment and identify ONLY genuine topic changes:

{formatted_content}

Context: {context if context else 'Podcast/video content'}

Mark timestamps only when the conversation genuinely shifts to a new topic or subject."""

    try:
        response = openai_client.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        if not response_content:
            logger.warning(f"Empty response for chunk {chunk_index + 1}")
            return []
        
        result = json.loads(response_content)
        timestamps_data = result.get("timestamps", [])
        
        formatted_timestamps = []
        for item in timestamps_data:
            time = item.get("time", "")
            description = item.get("description", "")
            
            if time and description:
                formatted_timestamps.append(f"{time} - {description}")
        
        logger.info(f"Generated {len(formatted_timestamps)} timestamps for chunk {chunk_index + 1}")
        return formatted_timestamps
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index + 1}: {str(e)}")
        return []