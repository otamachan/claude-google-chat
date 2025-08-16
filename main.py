"""
Google Chat Bot with Cloud Pub/Sub using async/await pattern.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import re

from google.apps import chat_v1 as google_chat
from google.cloud import pubsub_v1
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from claude_code_sdk import query as claude_query, ClaudeCodeOptions

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
)
logger = logging.getLogger(__name__)


class AsyncGoogleChatBot:
    """Async Google Chat Bot that processes messages from Pub/Sub."""
    
    def __init__(self):
        """Initialize the bot with credentials and configurations."""
        self.project_id = os.environ.get('PROJECT_ID')
        self.subscription_id = os.environ.get('SUBSCRIPTION_ID')
        self.service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        self.claude_api_key = os.environ.get('CLAUDE_API_KEY')
        
        # Validate required environment variables
        self._validate_env_vars()
        
        # Setup credentials and clients
        self._setup_clients()
        
        # Setup Claude client
        self._setup_claude_client()
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Track active tasks
        self.active_tasks = set()
        
    def _validate_env_vars(self):
        """Validate required environment variables."""
        required_vars = {
            'PROJECT_ID': self.project_id,
            'SUBSCRIPTION_ID': self.subscription_id,
            'GOOGLE_APPLICATION_CREDENTIALS': self.service_account_path
        }
        
        # Claude API key is optional
        if self.claude_api_key:
            logger.info('Claude API key found, Claude integration enabled')
        else:
            logger.warning('Claude API key not found, Claude integration disabled')
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            for var in missing_vars:
                logger.error(f'Missing {var} environment variable')
            sys.exit(1)
            
    def _setup_clients(self):
        """Setup Google Cloud clients."""
        # Setup credentials
        scopes = ['https://www.googleapis.com/auth/chat.bot']
        self.credentials = Credentials.from_service_account_file(
            self.service_account_path,
            scopes=scopes
        )
        
        # Setup Chat client
        self.chat_client = google_chat.ChatServiceClient(
            credentials=self.credentials
        )
        
        # Setup Pub/Sub client
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id, self.subscription_id
        )
    
    def _setup_claude_client(self):
        """Setup Claude Code SDK client."""
        if not self.claude_api_key:
            logger.warning('Claude API key not set, Claude features disabled')
            self.claude_enabled = False
            return
            
        try:
            # Set environment variable for Claude Code SDK
            os.environ['ANTHROPIC_API_KEY'] = self.claude_api_key
            self.claude_enabled = True
            logger.info('Claude Code SDK initialized successfully')
        except Exception as e:
            logger.error(f'Failed to initialize Claude Code SDK: {e}')
            self.claude_enabled = False

    async def start(self):
        """Start the bot and listen for messages."""
        logger.info(f'Starting bot, listening on {self.subscription_path}')
        
        try:
            await self._listen_for_messages()
        except KeyboardInterrupt:
            logger.info('Shutting down bot...')
            await self._shutdown()
        except Exception as e:
            logger.error(f'Unexpected error: {e}')
            await self._shutdown()
            
    async def _shutdown(self):
        """Gracefully shutdown the bot."""
        logger.info('Waiting for active tasks to complete...')
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        logger.info('Bot shutdown complete')

    async def _listen_for_messages(self):
        """Listen for messages from Pub/Sub subscription."""
        loop = asyncio.get_event_loop()
        
        def callback(message):
            """Callback for incoming Pub/Sub messages."""
            # Create async task for message handling using coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._handle_message(message), loop
            )
            # Track the task
            task = asyncio.wrap_future(future, loop=loop)
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.discard)
            
        # Configure flow control
        flow_control = pubsub_v1.types.FlowControl(
            max_messages=100,
            max_lease_duration=600  # 10 minutes
        )
        
        # Start subscription
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=callback,
            flow_control=flow_control
        )
        
        logger.info('Listening for messages...')
        
        # Keep the subscription alive
        with self.subscriber:
            try:
                # Run in executor to avoid blocking
                await loop.run_in_executor(
                    self.executor,
                    streaming_pull_future.result
                )
            except Exception as e:
                logger.error(f'Subscription error: {e}')
                streaming_pull_future.cancel()
                await streaming_pull_future.result()

    async def _handle_message(self, message):
        """Handle incoming Pub/Sub message."""
        try:
            # Parse message data
            event = json.loads(message.data.decode('utf-8'))
            logger.info(f'Received event: {event}')
            
            # Process based on event type
            response = await self._process_event(event)
            
            # Send response if needed
            if response:
                await self._send_response(response)
                
            # Acknowledge message
            message.ack()
            logger.info('Message processed and acknowledged')
            
        except json.JSONDecodeError as e:
            logger.error(f'Invalid JSON in message: {e}')
            message.ack()  # Ack to prevent redelivery of bad message
        except Exception as e:
            logger.error(f'Error processing message: {e}')
            message.nack()  # Nack for retry
            
    async def _process_event(self, event: Dict[str, Any]) -> Optional[google_chat.CreateMessageRequest]:
        """Process the event and generate appropriate response."""
        event_type = event.get('type')
        space_name = event.get('space', {}).get('name')
        
        if not space_name:
            logger.warning('Event missing space name')
            return None
            
        # Handle different event types
        if event_type == 'REMOVED_FROM_SPACE':
            logger.info(f'Bot removed from space: {space_name}')
            return None
            
        elif event_type == 'ADDED_TO_SPACE' and 'message' not in event:
            # Bot added via invite flow
            return self._create_welcome_message(space_name)
            
        elif event_type in ['ADDED_TO_SPACE', 'MESSAGE']:
            # Process message
            return await self._process_message(event, space_name)
            
        else:
            logger.warning(f'Unknown event type: {event_type}')
            return None
            
    def _create_welcome_message(self, space_name: str) -> google_chat.CreateMessageRequest:
        """Create welcome message when bot is added to space."""
        return google_chat.CreateMessageRequest(
            parent=space_name,
            message={
                'text': (
                    'こんにちは！Claude搭載のGoogle Chat Botです。\n'
                    '以下のコマンドが利用できます：\n'
                    '• `help` - ヘルプを表示\n'
                    '• `status` - ボットのステータス確認\n'
                    '• `time` - 現在時刻を表示\n'
                    '• `claude [質問]` - Claudeに質問\n'
                    '• その他のメッセージはClaudeが回答します'
                )
            }
        )

    async def _process_message(self, event: Dict[str, Any], space_name: str) -> Optional[google_chat.CreateMessageRequest]:
        """Process incoming message and generate response."""
        message = event.get('message', {})
        message_text = message.get('text', '').strip()
        thread_name = message.get('thread', {}).get('name')
        user_name = event.get('user', {}).get('displayName', 'Unknown')
        
        if not message_text:
            return None
            
        # Process command
        response_text = await self._handle_command(message_text, user_name)
        
        # Create response message
        return google_chat.CreateMessageRequest(
            parent=space_name,
            message_reply_option=google_chat.CreateMessageRequest.MessageReplyOption.REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD,
            message={
                'text': response_text,
                'thread': {'name': thread_name} if thread_name else {}
            }
        )

    async def _handle_command(self, text: str, user_name: str) -> str:
        """Handle user commands and generate response."""
        text_lower = text.lower()
        
        # Check for sleep command
        if text_lower.startswith('sleep'):
            return await self._handle_sleep_command(text)
        
        # Check for claude command or default to claude for all non-command messages
        if text_lower.startswith('claude '):
            # Remove claude prefix if present
            query = text[7:].strip()
            return await self._handle_claude_query(query, user_name)
        
        # Command handlers
        if text_lower in ['help', 'ヘルプ']:
            return self._get_help_text()
            
        elif text_lower in ['status', 'ステータス']:
            return await self._get_status()
            
        elif text_lower in ['time', '時刻', '時間']:
            return await self._get_current_time()
            
        elif any(greeting in text_lower for greeting in ['hello', 'hi', 'こんにちは', 'おはよう']):
            return f'こんにちは、{user_name}さん！何かお手伝いできることはありますか？'
            
        else:
            # Pass all other messages to Claude
            return await self._handle_claude_query(text, user_name)
    
    async def _handle_sleep_command(self, text: str) -> str:
        """Handle sleep command to simulate long-running tasks."""
        import re
        from datetime import datetime
        
        # Parse sleep duration from command
        match = re.search(r'sleep\s+(\d+)', text.lower())
        if match:
            duration = int(match.group(1))
            # Limit sleep duration to 30 seconds for safety
            duration = min(duration, 30)
            
            start_time = datetime.now()
            logger.info(f'Starting sleep for {duration} seconds at {start_time.strftime("%H:%M:%S")}...')
            
            # Immediately return processing message
            # Note: In a production system, you might want to send a follow-up message
            # after the sleep completes using a separate task
            processing_msg = f'⏳ {duration}秒間の処理を開始しました...\n開始時刻: {start_time.strftime("%H:%M:%S")}'
            
            # Simulate long-running task with asyncio.sleep
            await asyncio.sleep(duration)
            
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            logger.info(f'Sleep completed after {elapsed:.1f} seconds')
            
            # Return completion message with timing details
            return (
                f'✅ 処理が完了しました！\n'
                f'開始時刻: {start_time.strftime("%H:%M:%S")}\n'
                f'終了時刻: {end_time.strftime("%H:%M:%S")}\n'
                f'実行時間: {elapsed:.1f}秒'
            )
        else:
            return (
                'sleepコマンドの使い方: `sleep [秒数]` (例: sleep 5)\n'
                '最大30秒まで指定可能です。\n'
                '※ このコマンドは長時間タスクのシミュレーションに使用します。'
            )
    
    async def _handle_claude_query(self, query_text: str, user_name: str) -> str:
        """Handle queries to Claude API."""
        if not self.claude_enabled:
            return "❌ Claude APIが設定されていません。CLAUDE_API_KEYを確認してください。"
        
        try:
            logger.info(f'Sending query to Claude: {query_text[:100]}...')
            
            # Setup Claude Code options
            options = ClaudeCodeOptions(
                system_prompt="You are a helpful assistant in a Google Chat bot. Respond in Japanese when appropriate.",
                max_turns=1
            )
            
            # Collect response from Claude Code SDK
            response_parts = []
            raw_messages = []
            async for message in claude_query(prompt=query_text, options=options):
                # Store raw message for logging
                raw_messages.append(message)
                
                # Handle different message types from Claude Code SDK
                if hasattr(message, 'content'):
                    response_parts.append(str(message.content))
                elif hasattr(message, 'text'):
                    response_parts.append(str(message.text))
                else:
                    response_parts.append(str(message))
            
            response = ''.join(response_parts).strip()
            
            # Log raw messages in formatted way
            logger.info(f'Received {len(raw_messages)} messages from Claude')
            for i, msg in enumerate(raw_messages):
                formatted_msg = self._format_claude_message(msg)
                logger.info(f'Message {i+1}: {formatted_msg}')
            
            logger.info(f'Final response: {response}')
            
            # Format the response
            if response:
                return f"🤖 Claude:\n{response}"
            else:
                return "❌ Claudeから応答がありませんでした。"
                
        except Exception as e:
            logger.error(f'Error calling Claude API: {e}')
            return f"❌ エラーが発生しました: {str(e)}"
            
    def _get_help_text(self) -> str:
        """Get help text."""
        claude_status = "✅ 有効" if self.claude_enabled else "❌ 無効 (CLAUDE_API_KEYが未設定)"
        
        return (
            '利用可能なコマンド：\n'
            '• `help` - このヘルプメッセージを表示\n'
            '• `status` - ボットのステータスを確認\n'
            '• `time` - 現在時刻を表示\n'
            '• `sleep [秒数]` - 指定秒数待機（最大30秒）\n'
            '• `claude [質問]` - Claudeに質問\n'
            '• `hello` / `hi` - 挨拶\n'
            '• その他のメッセージ - Claudeが回答\n\n'
            f'Claude統合: {claude_status}\n\n'
            '※ コマンド以外のメッセージは自動的にClaudeに送信されます'
        )
        
    async def _get_status(self) -> str:
        """Get bot status."""
        active_tasks_count = len(self.active_tasks)
        claude_status = "✅ 有効" if self.claude_enabled else "❌ 無効"
        
        return (
            f'✅ ボットは正常に動作しています\n'
            f'アクティブなタスク数: {active_tasks_count}\n'
            f'プロジェクトID: {self.project_id}\n'
            f'サブスクリプション: {self.subscription_id}\n'
            f'Claude統合: {claude_status}'
        )
        
    async def _get_current_time(self) -> str:
        """Get current time."""
        from datetime import datetime
        import pytz
        
        # Get JST time
        jst = pytz.timezone('Asia/Tokyo')
        current_time = datetime.now(jst)
        
        return f'現在時刻（JST）: {current_time.strftime("%Y年%m月%d日 %H:%M:%S")}'
    
    def _format_claude_message(self, message) -> str:
        """Format Claude message for better readability."""
        try:
            # Try to convert to dict if it has a dict representation
            if hasattr(message, '__dict__'):
                msg_dict = message.__dict__
            elif hasattr(message, '_asdict'):
                msg_dict = message._asdict()
            else:
                msg_dict = {'raw_message': str(message)}
            
            # Pretty format the dictionary
            formatted_parts = []
            for key, value in msg_dict.items():
                if key.startswith('_'):
                    continue
                    
                if isinstance(value, str) and len(value) > 100:
                    formatted_parts.append(f'{key}: {value[:100]}...')
                else:
                    formatted_parts.append(f'{key}: {value}')
            
            return '{ ' + ', '.join(formatted_parts) + ' }'
            
        except Exception as e:
            return f'Error formatting message: {e} | Raw: {str(message)}'

    async def _send_response(self, request: google_chat.CreateMessageRequest):
        """Send response message to Google Chat."""
        loop = asyncio.get_event_loop()
        try:
            # Run blocking call in executor
            await loop.run_in_executor(
                self.executor,
                self.chat_client.create_message,
                request
            )
            logger.info('Response sent successfully')
        except Exception as e:
            logger.error(f'Failed to send response: {e}')


async def main():
    """Main entry point."""
    bot = AsyncGoogleChatBot()
    await bot.start()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Bot stopped by user')
    except Exception as e:
        logger.error(f'Fatal error: {e}')
        sys.exit(1)