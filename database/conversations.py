"""Module for managing conversation histories in a simple database"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, message_to_dict, messages_from_dict

class ConversationStore:
    """Store for managing conversation histories"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the conversation store
        
        Args:
            storage_path: Path to store conversation data (defaults to 'conversations' in current dir)
        """
        self.storage_path = storage_path or os.path.join(os.getcwd(), "conversations")
        os.makedirs(self.storage_path, exist_ok=True)
        
    def _message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert a message to a serializable dict"""
        return message_to_dict(message)

    def _dict_to_message(self, message_dict: Dict[str, Any]) -> BaseMessage:
        """Convert a dict back to a message"""
        messages = messages_from_dict([message_dict])
        return messages[0] if messages else None
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation
        
        Args:
            title: Optional title for the conversation
            
        Returns:
            str: Conversation ID
        """
        conv_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        conversation = {
            "id": conv_id,
            "title": title or f"Conversation {timestamp}",
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": []
        }
        
        self._save_conversation(conv_id, conversation)
        return conv_id
    
    def save_messages(self, conv_id: str, messages: List[BaseMessage]) -> bool:
        """Save messages to a conversation
        
        Args:
            conv_id: Conversation ID
            messages: List of messages to save
            
        Returns:
            bool: Success status
        """
        try:
            conversation = self._load_conversation(conv_id)
            if not conversation:
                return False
                
            # Convert messages to dicts
            message_dicts = [self._message_to_dict(msg) for msg in messages]
            
            # Update conversation
            conversation["messages"] = message_dicts
            conversation["updated_at"] = datetime.now().isoformat()
            
            # Save back to storage
            self._save_conversation(conv_id, conversation)
            return True
        except Exception as e:
            print(f"Error saving messages: {e}")
            return False
    
    def load_messages(self, conv_id: str) -> List[BaseMessage]:
        """Load messages from a conversation
        
        Args:
            conv_id: Conversation ID
            
        Returns:
            List[BaseMessage]: List of messages
        """
        conversation = self._load_conversation(conv_id)
        if not conversation:
            return []
            
        # Convert dicts back to messages
        return [self._dict_to_message(msg) for msg in conversation.get("messages", [])]
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations
        
        Returns:
            List[Dict[str, Any]]: List of conversation metadata
        """
        conversations = []
        
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.storage_path, filename)
                    with open(file_path, 'r') as f:
                        conversation = json.load(f)
                        # Include only metadata, not messages
                        conversations.append({
                            "id": conversation.get("id"),
                            "title": conversation.get("title"),
                            "created_at": conversation.get("created_at"),
                            "updated_at": conversation.get("updated_at"),
                            "message_count": len(conversation.get("messages", []))
                        })
        except Exception as e:
            print(f"Error listing conversations: {e}")
        
        # Sort by updated_at (most recent first)
        return sorted(conversations, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    def update_conversation_title(self, conv_id: str, title: str) -> bool:
        """Update conversation title
        
        Args:
            conv_id: Conversation ID
            title: New title
            
        Returns:
            bool: Success status
        """
        try:
            conversation = self._load_conversation(conv_id)
            if not conversation:
                return False
                
            conversation["title"] = title
            conversation["updated_at"] = datetime.now().isoformat()
            
            self._save_conversation(conv_id, conversation)
            return True
        except Exception as e:
            print(f"Error updating conversation title: {e}")
            return False
    
    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation
        
        Args:
            conv_id: Conversation ID
            
        Returns:
            bool: Success status
        """
        try:
            file_path = os.path.join(self.storage_path, f"{conv_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False
    
    def _save_conversation(self, conv_id: str, conversation: Dict[str, Any]) -> None:
        """Save conversation to file"""
        try:
            file_path = os.path.join(self.storage_path, f"{conv_id}.json")
            with open(file_path, 'w') as f:
                json.dump(conversation, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation to file: {e}")
    
    def _load_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation from file"""
        try:
            file_path = os.path.join(self.storage_path, f"{conv_id}.json")
            if not os.path.exists(file_path):
                return None
                
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading conversation from file: {e}")
            return None

# Initialize global conversation store
conversation_store = ConversationStore()
