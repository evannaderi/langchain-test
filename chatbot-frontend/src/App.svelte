<script>
  import { onMount } from 'svelte';
  import Settings from './Settings.svelte';
  import ChatWindow from './ChatWindow.svelte';

  let settings = {
    language: 'English',
    speaking_level: 'intermediate',
    tone: 'friendly',
    specific_words: '',
    additional_instructions: ''
  };

  let messages = [];
  let inputMessage = '';
  let showSettings = false;

  function toggleSettings() {
    showSettings = !showSettings;
  }

  function updateSettings(newSettings) {
    settings = newSettings;
    showSettings = false;
  }

  async function sendMessage() {
    if (inputMessage.trim() === '') return;

    messages = [...messages, { role: 'user', content: inputMessage }];
    const userMessage = inputMessage;
    inputMessage = '';

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': 'some-unique-session-id'  // You can generate this dynamically if needed
        },
        body: JSON.stringify({ message: userMessage, settings }),
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const reader = response.body.getReader();
      let botReply = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        botReply += new TextDecoder().decode(value);
        messages = [...messages.slice(0, -1), { role: 'assistant', content: botReply }];
      }

    } catch (error) {
      console.error('Error:', error);
      messages = [...messages, { role: 'error', content: 'An error occurred. Please try again.' }];
    }
  }
</script>

<main>
  <h1>Customizable Chatbot</h1>
  <button on:click={toggleSettings}>
    {showSettings ? 'Close Settings' : 'Open Settings'}
  </button>

  {#if showSettings}
    <Settings {settings} on:update={event => updateSettings(event.detail)} />
  {/if}

  <ChatWindow {messages} />

  <div class="input-area">
    <input 
      type="text" 
      bind:value={inputMessage} 
      on:keypress={event => event.key === 'Enter' && sendMessage()}
      placeholder="Type your message here..."
    />
    <button on:click={sendMessage}>Send</button>
  </div>
</main>

<style>
  main {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
  }

  .input-area {
    display: flex;
    margin-top: 20px;
  }

  input {
    flex-grow: 1;
    padding: 10px;
    font-size: 16px;
  }

  button {
    padding: 10px 20px;
    font-size: 16px;
    background-color: #4CAF50;
    color: white;
    border: none;
    cursor: pointer;
  }

  button:hover {
    background-color: #45a049;
  }
</style>
