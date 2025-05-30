import os
import json
import logging
from typing import List, Dict, Tuple
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


def calculate_chunk_size(srt_entries: List[Dict[str, str]]) -> int:
    """
    Calculate optimal chunk size based on content length to stay within token limits.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries
        
    Returns:
        int: Optimal chunk size (number of entries per chunk)
    """
    total_entries = len(srt_entries)
    
    # Estimate average characters per entry (including formatting)
    sample_size = min(10, total_entries)
    total_chars = sum(len(entry['text']) + len(entry['start']) + 10 for entry in srt_entries[:sample_size])
    avg_chars_per_entry = total_chars / sample_size if sample_size > 0 else 100
    
    # Conservative token estimate: ~4 chars per token, target ~3000 tokens per chunk
    # This leaves room for system prompt and response
    target_chars_per_chunk = 12000
    chunk_size = max(10, int(target_chars_per_chunk / avg_chars_per_entry))
    
    # Cap chunk size to reasonable limits
    chunk_size = min(chunk_size, max(50, total_entries // 4))
    
    logger.debug(f"Calculated chunk size: {chunk_size} entries (avg {avg_chars_per_entry:.1f} chars/entry)")
    return chunk_size


def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS.mmm format to seconds."""
    try:
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    except:
        return 0.0


def create_smart_chunks(srt_entries: List[Dict[str, str]], chunk_size: int) -> List[Tuple[List[Dict[str, str]], int, int]]:
    """
    Create smart chunks with overlap, preferring sentence boundaries and natural breaks.
    
    Args:
        srt_entries (List[Dict[str, str]]): List of SRT entries
        chunk_size (int): Target size for each chunk
        
    Returns:
        List[Tuple[List[Dict[str, str]], int, int]]: List of (chunk_entries, start_idx, end_idx)
    """
    if len(srt_entries) <= chunk_size:
        return [(srt_entries, 0, len(srt_entries) - 1)]
    
    chunks = []
    overlap_size = max(2, chunk_size // 10)  # 10% overlap, minimum 2 entries
    i = 0
    
    while i < len(srt_entries):
        # Determine chunk end
        chunk_end = min(i + chunk_size, len(srt_entries))
        
        # Try to find a better break point near the end of the chunk
        if chunk_end < len(srt_entries):
            # Look for sentence endings within the last 20% of the chunk
            search_start = max(i + int(chunk_size * 0.8), i + 1)
            search_end = min(chunk_end + 5, len(srt_entries))
            
            best_break = chunk_end
            for j in range(search_start, search_end):
                if j >= len(srt_entries):
                    break
                    
                text = srt_entries[j]['text'].strip()
                # Look for sentence endings
                if text.endswith(('.', '!', '?', '。', '！', '？')):
                    best_break = j + 1
                    break
                
                # Look for significant time gaps (potential topic breaks)
                if j > 0:
                    try:
                        current_time = time_to_seconds(srt_entries[j]['start'])
                        prev_time = time_to_seconds(srt_entries[j-1]['end'])
                        if current_time - prev_time > 3.0:  # 3+ second gap
                            best_break = j
                            break
                    except:
                        pass
            
            chunk_end = best_break
        
        # Create chunk
        chunk_entries = srt_entries[i:chunk_end]
        chunks.append((chunk_entries, i, chunk_end - 1))
        
        # Move to next chunk with overlap
        if chunk_end >= len(srt_entries):
            break
        i = max(chunk_end - overlap_size, i + 1)
    
    logger.debug(f"Created {len(chunks)} chunks with overlap")
    return chunks


def merge_chunk_results(chunk_results: List[List[str]], chunk_info: List[Tuple[List[Dict[str, str]], int, int]]) -> List[str]:
    """
    Merge timestamp results from multiple chunks, removing duplicates and sorting.
    
    Args:
        chunk_results (List[List[str]]): Results from each chunk
        chunk_info (List[Tuple]): Information about each chunk
        
    Returns:
        List[str]: Merged and deduplicated timestamps
    """
    all_timestamps = []
    seen_times = set()
    
    for chunk_idx, timestamps in enumerate(chunk_results):
        for timestamp in timestamps:
            # Extract time from timestamp (format: "H:MM:SS - Description")
            if ' - ' in timestamp:
                time_part = timestamp.split(' - ')[0].strip()
                
                # Avoid near-duplicate timestamps (within 10 seconds)
                time_seconds = time_to_seconds(time_part + '.000')
                is_duplicate = any(abs(time_seconds - seen_time) < 10 for seen_time in seen_times)
                
                if not is_duplicate:
                    all_timestamps.append(timestamp)
                    seen_times.add(time_seconds)
    
    # Sort by timestamp
    def sort_key(ts):
        try:
            time_part = ts.split(' - ')[0].strip()
            return time_to_seconds(time_part + '.000')
        except:
            return 0
    
    all_timestamps.sort(key=sort_key)
    logger.debug(f"Merged {len(all_timestamps)} unique timestamps from {len(chunk_results)} chunks")
    return all_timestamps

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
- Descriptions should be brief but descriptive and detailed as possible. Use overall show context to get idea of whats being talked about. (under 115 characters)
- Focus on meaningful content transitions, not minor topic shifts
- Number of segments should be reasonable (around 5-15 per hour of content)
- Start with "0:00:00" if the content begins immediately

Example output format:
{
  "timestamps": [
    {"time": "0:00:00", "description": "Woody introduces the show and guest, Taylor the sponsors - New mystery sponsor!"},
    {"time": "0:02:15", "description": "Why Ed would never do a ‘Top Gear’ style show & how brand deals ruined content"},
    {"time": "0:15:30", "description": "Movie/TV talk: Tom Cruise, Yellowstone and why every actor is CGI deaged in 2025"},
    {"time": "0:15:30", "description": "Demi Moore’s huge bush, female body hair trends & PKA’s hairy legs"},
    {"time": "0:28:45", "description": "The guys call it a show"}
  ]
}"""

        # Build the user prompt
        user_prompt = f"Analyze the following subtitle content and identify topic segments:\n\n{formatted_content}"
        
        if context:
            user_prompt += f"\n\nAdditional context: {context}"
        
        user_prompt += "\n\nGenerate topic-based timestamps in JSON format as specified."
        
        logger.debug(f"Sending request to OpenAI with {len(srt_entries)} subtitle entries")
        
        # Make API call to OpenAI with extended timeout
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=10000,
            response_format={"type": "json_object"},
            timeout=120  # 2 minutes timeout for large files
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
