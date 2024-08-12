<script>
  import { createEventDispatcher } from 'svelte';

  export let settings;

  const dispatch = createEventDispatcher();

  function updateSettings() {
    dispatch('update', settings);
  }
</script>

<div class="settings">
  <h2>Chatbot Settings</h2>
  <label>
    Language:
    <input type="text" bind:value={settings.language}>
  </label>
  <label>
    Speaking Level:
    <select bind:value={settings.speaking_level}>
      <option value="beginner">Beginner</option>
      <option value="intermediate">Intermediate</option>
      <option value="advanced">Advanced</option>
    </select>
  </label>
  <label>
    Tone:
    <input type="text" bind:value={settings.tone}>
  </label>
  <label>
    Specific Words:
    <input type="text" bind:value={settings.specific_words}>
  </label>
  <label>
    Additional Instructions:
    <textarea bind:value={settings.additional_instructions}></textarea>
  </label>
  <button on:click={updateSettings}>Apply Settings</button>
</div>

<style>
  .settings {
    background-color: #f0f0f0;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 5px;
  }

  label {
    display: block;
    margin-bottom: 10px;
  }

  input, select, textarea {
    width: 100%;
    padding: 5px;
    font-size: 14px;
  }

  textarea {
    height: 100px;
  }

  button {
    margin-top: 10px;
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
