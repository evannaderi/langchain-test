<script>
  import { afterUpdate } from 'svelte';

  export let messages = [];

  let chatWindow;

  afterUpdate(() => {
    if (chatWindow) {
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }
  });
</script>

<div class="chat-window" bind:this={chatWindow}>
  {#each messages as message}
    <div class="message {message.role}">
      <strong>{message.role === 'user' ? 'You' : 'Bot'}:</strong>
      <p>{message.content}</p>
    </div>
  {/each}
</div>

<style>
  .chat-window {
    height: 400px;
    overflow-y: auto;
    border: 1px solid #ccc;
    padding: 10px;
    margin-bottom: 20px;
  }

  .message {
    margin-bottom: 10px;
    padding: 5px;
    border-radius: 5px;
  }

  .user {
    background-color: #e6f2ff;
  }

  .assistant {
    background-color: #f0f0f0;
  }

  .error {
    background-color: #ffcccb;
  }
</style>
