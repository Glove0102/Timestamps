import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class SRTParseError(Exception):
    """Custom exception for SRT parsing errors"""
    pass

def parse_srt_file(srt_content: str) -> List[Dict[str, str]]:
    """
    Parse SRT file content and return list of subtitle entries.
    
    Args:
        srt_content (str): The raw SRT file content
        
    Returns:
        List[Dict[str, str]]: List of subtitle entries with 'start', 'end', and 'text' keys
        
    Raises:
        SRTParseError: If the SRT file format is invalid or no entries are found
    """
    try:
        # Remove BOM if present
        if srt_content.startswith('\ufeff'):
            srt_content = srt_content[1:]
        
        # Split content into blocks (entries are separated by double newlines)
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        
        if not blocks:
            raise SRTParseError("No content found in SRT file")
        
        entries = []
        
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
                
            lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
            
            if len(lines) < 3:
                logger.warning(f"Skipping malformed block {i+1}: insufficient lines")
                continue
            
            # First line should be the sequence number
            try:
                sequence_num = int(lines[0])
            except ValueError:
                logger.warning(f"Skipping block {i+1}: invalid sequence number '{lines[0]}'")
                continue
            
            # Second line should be the timestamp
            timestamp_line = lines[1]
            if not re.match(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', timestamp_line):
                logger.warning(f"Skipping block {i+1}: invalid timestamp format '{timestamp_line}'")
                continue
            
            # Parse start and end times
            try:
                times = timestamp_line.split('-->')
                start_time = times[0].strip().replace(',', '.')
                end_time = times[1].strip().replace(',', '.')
                
                # Convert to standard format (HH:MM:SS)
                start_time = convert_to_standard_time(start_time)
                end_time = convert_to_standard_time(end_time)
                
            except Exception as e:
                logger.warning(f"Skipping block {i+1}: error parsing timestamps - {str(e)}")
                continue
            
            # Remaining lines are the subtitle text
            subtitle_text = ' '.join(lines[2:])
            
            # Clean up HTML tags and formatting
            subtitle_text = clean_subtitle_text(subtitle_text)
            
            if subtitle_text:  # Only add if there's actual text content
                entries.append({
                    'start': start_time,
                    'end': end_time,
                    'text': subtitle_text,
                    'sequence': sequence_num
                })
        
        if not entries:
            raise SRTParseError("No valid subtitle entries found in SRT file")
        
        logger.info(f"Successfully parsed {len(entries)} subtitle entries")
        return entries
    
    except SRTParseError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing SRT file: {str(e)}")
        raise SRTParseError(f"Failed to parse SRT file: {str(e)}")

def convert_to_standard_time(time_str: str) -> str:
    """
    Convert SRT timestamp to standard HH:MM:SS format.
    
    Args:
        time_str (str): Time string in format HH:MM:SS.mmm
        
    Returns:
        str: Time string in format HH:MM:SS
    """
    try:
        # Remove milliseconds part
        if '.' in time_str:
            time_str = time_str.split('.')[0]
        
        # Validate format
        if not re.match(r'\d{2}:\d{2}:\d{2}', time_str):
            raise ValueError(f"Invalid time format: {time_str}")
        
        return time_str
    except Exception as e:
        logger.error(f"Error converting time format: {str(e)}")
        raise

def clean_subtitle_text(text: str) -> str:
    """
    Clean subtitle text by removing HTML tags and extra whitespace.
    
    Args:
        text (str): Raw subtitle text
        
    Returns:
        str: Cleaned subtitle text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove formatting tags like {\\i1}, {\\b1}, etc.
    text = re.sub(r'\{[^}]+\}', '', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    return text.strip()

def format_srt_for_analysis(entries: List[Dict[str, str]]) -> str:
    """
    Format SRT entries for OpenAI analysis.
    
    Args:
        entries (List[Dict[str, str]]): List of SRT entries
        
    Returns:
        str: Formatted text for analysis
    """
    formatted_lines = []
    
    for entry in entries:
        formatted_lines.append(f"[{entry['start']}] {entry['text']}")
    
    return '\n'.join(formatted_lines)
