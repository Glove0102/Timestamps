import os
import json
import logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
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
- Use the format "H:MM:SS" for timestamps (e.g., "0:00:15", "0:18:11", "1:23:45")
- Descriptions should be brief but descriptive (80-140 characters)
- Focus on meaningful content transitions, not minor topic shifts
- Include 8-12 topic segments per hour of content
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
            model="o4-mini-2025-04-16",
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

def parse_srt_time_to_seconds(time_str: str) -> int:
    """
    Convert SRT timestamp to seconds.
    
    Args:
        time_str (str): Time string in format HH:MM:SS.mmm or HH:MM:SS
        
    Returns:
        int: Time in total seconds
    """
    try:
        # Remove milliseconds if present (format: HH:MM:SS.mmm or HH:MM:SS,mmm)
        if '.' in time_str:
            time_str = time_str.split('.')[0]
        elif ',' in time_str:
            time_str = time_str.split(',')[0]
        
        # Parse HH:MM:SS
        time_parts = time_str.split(':')
        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            logger.warning(f"Invalid time format: {time_str}")
            return 0
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse time '{time_str}': {e}")
        return 0

def get_video_duration_minutes(srt_entries: List[Dict[str, str]]) -> float:
    """
    Get total video duration in minutes from SRT entries.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries
        
    Returns:
        float: Video duration in minutes
    """
    if not srt_entries:
        return 0
    
    # Get the end time of the last entry
    last_entry = srt_entries[-1]
    last_end_seconds = parse_srt_time_to_seconds(last_entry['end'])
    return last_end_seconds / 60.0

def chunk_srt_entries(srt_entries: List[Dict[str, str]], chunk_duration_minutes: int = 60) -> List[List[Dict[str, str]]]:
    """
    Split SRT entries into chunks based on duration.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries
        chunk_duration_minutes (int): Duration of each chunk in minutes
        
    Returns:
        List[List[Dict[str, str]]]: List of SRT entry chunks
    """
    if not srt_entries:
        return []
    
    chunks = []
    current_chunk = []
    chunk_duration_seconds = chunk_duration_minutes * 60
    
    # Start from the beginning of the video
    first_entry_start = parse_srt_time_to_seconds(srt_entries[0]['start'])
    chunk_end_time = first_entry_start + chunk_duration_seconds
    
    for entry in srt_entries:
        entry_start_seconds = parse_srt_time_to_seconds(entry['start'])
        
        # If this entry starts after the current chunk's end time, finalize the chunk
        if entry_start_seconds >= chunk_end_time and current_chunk:
            chunks.append(current_chunk)
            logger.debug(f"Created chunk with {len(current_chunk)} entries, duration: {(parse_srt_time_to_seconds(current_chunk[-1]['end']) - parse_srt_time_to_seconds(current_chunk[0]['start']))/60:.1f} minutes")
            current_chunk = []
            # Set the next chunk end time
            chunk_end_time = entry_start_seconds + chunk_duration_seconds
        
        current_chunk.append(entry)
    
    # Add the last chunk if it has entries
    if current_chunk:
        chunks.append(current_chunk)
        logger.debug(f"Created final chunk with {len(current_chunk)} entries")
    
    logger.info(f"Split {len(srt_entries)} entries into {len(chunks)} chunks")
    return chunks

def generate_topic_timestamps_for_long_video(srt_entries: List[Dict[str, str]], context: str = None, chunk_duration_minutes: int = 60) -> List[str]:
    """
    Generate topic-based timestamps for long videos using chunking approach.
    This function is used for videos over 2 hours long.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries with start, end, text
        context (str, optional): Additional context for better topic detection
        chunk_duration_minutes (int): Duration of each chunk in minutes (default: 60)
        
    Returns:
        List[str]: List of formatted timestamps with topic descriptions
        
    Raises:
        OpenAIServiceError: If OpenAI API call fails or returns invalid response
    """
    if not openai_client:
        raise OpenAIServiceError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
    
    video_duration = get_video_duration_minutes(srt_entries)
    logger.info(f"Processing long video: {video_duration:.1f} minutes duration with {len(srt_entries)} subtitle entries")
    
    # Split into chunks
    chunks = chunk_srt_entries(srt_entries, chunk_duration_minutes)
    logger.info(f"Split video into {len(chunks)} chunks of ~{chunk_duration_minutes} minutes each")
    
    all_timestamps = []
    
    for chunk_index, chunk in enumerate(chunks):
        chunk_start_time = parse_srt_time_to_seconds(chunk[0]['start']) if chunk else 0
        chunk_end_time = parse_srt_time_to_seconds(chunk[-1]['end']) if chunk else 0
        chunk_duration = (chunk_end_time - chunk_start_time) / 60.0
        
        logger.info(f"Processing chunk {chunk_index + 1}/{len(chunks)}: {chunk_duration:.1f} minutes")
        
        try:
            # Format chunk content for analysis
            formatted_content = format_srt_for_openai(chunk)
            
            # Adjust system prompt for chunk processing
            topics_per_hour = 8  # Base rate - more conservative
            expected_topics = max(3, int(topics_per_hour * chunk_duration / 60))
            
            system_prompt = f"""You are an expert at analyzing video content and identifying topic segments from subtitles. 
Your task is to analyze subtitle text with timestamps and identify distinct topic segments or content changes.

This is chunk {chunk_index + 1} of {len(chunks)} from a longer video. You MUST identify at least {expected_topics} meaningful topic segments within this chunk.

Return a JSON object with a "timestamps" array containing objects with "time" and "description" fields.
Each timestamp should mark the beginning of a new topic or significant content shift.

CRITICAL REQUIREMENTS:
- You MUST generate at least {expected_topics} timestamps for this chunk
- Use timestamps that appear in the provided subtitle content
- Use the EXACT format "H:MM:SS" for timestamps (e.g., "0:15:30", "1:18:11", "2:23:45")
- Look for natural conversation breaks, topic changes, or content shifts
- Include timestamps from the beginning, middle, and end of the chunk
- Descriptions should be 60-120 characters and describe what's happening

Example output format:
{{
  "timestamps": [
    {{"time": "1:15:30", "description": "Discussion about technical details and implementation"}},
    {{"time": "1:28:45", "description": "Audience questions and answers session"}},
    {{"time": "1:42:15", "description": "Moving to next major topic or segment"}}
  ]
}}"""

            # Build the user prompt for this chunk
            chunk_start_time_str = chunk[0]['start'] if chunk else "0:00:00"
            chunk_end_time_str = chunk[-1]['end'] if chunk else "0:00:00"
            
            user_prompt = f"Analyze the following subtitle content chunk (from {chunk_start_time_str} to {chunk_end_time_str}) and identify topic segments:\n\n{formatted_content}"
            
            if context:
                user_prompt += f"\n\nAdditional context: {context}"
            
            user_prompt += f"\n\nYou must generate at least {expected_topics} timestamps for this chunk. This is chunk {chunk_index + 1} of {len(chunks)} from a {video_duration:.0f}-minute video. Focus on identifying natural breaks and topic transitions within this specific time range."
            
            logger.debug(f"Sending chunk {chunk_index + 1} to OpenAI with {len(chunk)} subtitle entries")
            
            # Make API call to OpenAI for this chunk
            response = openai_client.chat.completions.create(
                model="o4-mini-2025-04-16",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            response_content = response.choices[0].message.content
            if not response_content:
                logger.warning(f"Empty response from OpenAI for chunk {chunk_index + 1}")
                continue
                
            logger.debug(f"Received response from OpenAI for chunk {chunk_index + 1}: {response_content[:200]}...")
            
            try:
                result = json.loads(response_content)
                timestamps_data = result.get("timestamps", [])
                
                # Format timestamps for this chunk
                chunk_timestamps = []
                for item in timestamps_data:
                    time = item.get("time", "")
                    description = item.get("description", "")
                    
                    if time and description:
                        chunk_timestamps.append(f"{time} - {description}")
                    else:
                        logger.warning(f"Skipping malformed timestamp entry in chunk {chunk_index + 1}: {item}")
                
                if chunk_timestamps:
                    all_timestamps.extend(chunk_timestamps)
                    logger.info(f"Chunk {chunk_index + 1} generated {len(chunk_timestamps)} timestamps: {[ts.split(' - ')[0] for ts in chunk_timestamps]}")
                else:
                    logger.warning(f"No valid timestamps generated for chunk {chunk_index + 1}")
                    logger.debug(f"Raw OpenAI response for failed chunk {chunk_index + 1}: {response_content}")
                    
                    # Try to extract any timestamps from the raw response as fallback
                    import re
                    time_pattern = r'\d{1,2}:\d{2}:\d{2}'
                    found_times = re.findall(time_pattern, response_content)
                    if found_times:
                        logger.info(f"Found fallback timestamps in chunk {chunk_index + 1}: {found_times}")
                        for time_stamp in found_times[:expected_topics]:
                            fallback_desc = f"Topic segment at {time_stamp}"
                            all_timestamps.append(f"{time_stamp} - {fallback_desc}")
                    else:
                        logger.error(f"No timestamps found at all in chunk {chunk_index + 1} response")
            
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI JSON response for chunk {chunk_index + 1}: {str(e)}")
                continue
        
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index + 1}: {str(e)}")
            continue
    
    if not all_timestamps:
        raise OpenAIServiceError("No timestamps could be generated for any chunks")
    
    # Sort timestamps by time to ensure proper order
    def safe_sort_key(timestamp_str):
        try:
            time_part = timestamp_str.split(' - ')[0]
            return parse_srt_time_to_seconds(time_part)
        except (IndexError, ValueError):
            logger.warning(f"Failed to parse timestamp for sorting: {timestamp_str}")
            return 0
    
    all_timestamps.sort(key=safe_sort_key)
    
    logger.info(f"Successfully generated {len(all_timestamps)} total topic timestamps from {len(chunks)} chunks")
    return all_timestamps
