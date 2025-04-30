#!/usr/bin/env python3
"""
Extract Content from AI Response Messages

This script demonstrates how to extract just the content from response messages
in a conversation with an AI model using MCP and tools.
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class MessageBase:
    """Base class for message types"""
    content: str
    id: str
    
@dataclass
class HumanMessage(MessageBase):
    """Human message in a conversation"""
    pass

@dataclass
class AIMessage(MessageBase):
    """AI message in a conversation"""
    tool_calls: List[Dict[str, Any]] = None
    
@dataclass
class ToolMessage(MessageBase):
    """Tool message in a conversation"""
    name: str
    tool_call_id: str

def parse_response(response_dict):
    """
    Parse a response dictionary and extract the content from messages.
    
    Args:
        response_dict: Dictionary containing response data
        
    Returns:
        List of extracted message contents
    """
    messages = response_dict.get('messages', [])
    extracted_content = []
    
    for msg in messages:
        # Determine message type from structure
        if isinstance(msg, dict):
            msg_type = msg.get('type', 'unknown')
        else:
            # This is for object instances where we look at the class name
            msg_type = msg.__class__.__name__
        
        # Create a dictionary with message information
        message_info = {
            'type': msg_type,
            'content': getattr(msg, 'content', msg.get('content', '')) if msg else ''
        }
        
        # Add tool information if available
        if hasattr(msg, 'name') and msg.name:
            message_info['tool_name'] = msg.name
            
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            message_info['tool_calls'] = msg.tool_calls
            
        extracted_content.append(message_info)
    
    return extracted_content

def main():
    """
    Example of using the parse_response function on a sample response
    """
    # Sample response data (similar to what was in the prompt)
    response_data = {
        'messages': [
            # These would be actual message objects in real code
            # Here we're simulating the structure as dictionaries for demonstration
            {
                'type': 'HumanMessage',
                'content': "what's (3 + 5) x 12?",
                'id': 'df2fa29d-a3b2-45c5-895f-ba39c004e640'
            },
            {
                'type': 'AIMessage',
                'content': '',
                'id': 'run-bf5f8292-faa9-4608-aef3-08dc7394f15e-0',
                'tool_calls': [{'name': 'add', 'args': {'a': '3', 'b': '5'}, 'id': '7133f50c-e741-470f-a63e-c1b368c19c06', 'type': 'tool_call'}]
            },
            {
                'type': 'ToolMessage',
                'content': '8',
                'name': 'add',
                'id': '1f233610-5bac-4814-9c8b-35ca1fcbb3c9',
                'tool_call_id': '7133f50c-e741-470f-a63e-c1b368c19c06'
            },
            {
                'type': 'AIMessage',
                'content': 'To calculate the expression `(3 + 5) * 12`, we first need to evaluate the addition inside the parentheses, which gives us `8`. Then we multiply `8` by `12`, resulting in `96`. Therefore, the final answer is `96`.',
                'id': 'run-170aaf43-755d-4b5e-a253-aba12494d507-0'
            }
        ]
    }
    
    # Extract content from the response
    extracted = parse_response(response_data)
    
    # Print the extracted content
    print("Extracted message content:")
    print("-" * 50)
    
    for i, msg in enumerate(extracted):
        print(f"Message {i+1} ({msg['type']}):")
        
        # Print content if it exists
        if msg['content']:
            print(f"Content: {msg['content']}")
        
        # Print tool information if it exists
        if 'tool_name' in msg:
            print(f"Tool: {msg['tool_name']}")
            
        if 'tool_calls' in msg:
            for call in msg['tool_calls']:
                print(f"Tool Call: {call['name']} with args {call['args']}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()