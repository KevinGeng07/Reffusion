<script setup lang="ts">
import { computed, ref } from "vue";
import { login, signup } from "../api";

const emit = defineEmits<{
  success: [];
}>();

const mode = ref<"sign-in" | "sign-up">("sign-in");
const username = ref("");
const password = ref("");
const error = ref("");
const busy = ref(false);

const isSignUp = computed(() => mode.value === "sign-up");

function toggleMode() {
  mode.value = isSignUp.value ? "sign-in" : "sign-up";
  error.value = "";
}

async function handleSubmit() {
  busy.value = true;
  error.value = "";
  try {
    if (isSignUp.value) {
      await signup(username.value.trim(), password.value);
    } else {
      await login(username.value.trim(), password.value);
    }
    emit("success");
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not sign in.";
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="login">
    <form class="login__card" @submit.prevent="handleSubmit">
      <h1 class="login__title">Reffusion</h1>
      <p class="login__subtitle">
        {{ isSignUp ? "Create an account." : "Sign in to your account." }}
      </p>

      <label class="login__field">
        <span>Username</span>
        <input v-model="username" type="text" autocomplete="username" required autofocus />
      </label>

      <label class="login__field">
        <span>Password</span>
        <input
          v-model="password"
          type="password"
          :autocomplete="isSignUp ? 'new-password' : 'current-password'"
          required
        />
      </label>

      <p v-if="error" class="login__error">{{ error }}</p>

      <button class="login__submit" type="submit" :disabled="busy">
        {{ busy ? "Working…" : isSignUp ? "Create account" : "Sign in" }}
      </button>

      <button class="login__toggle" type="button" @click="toggleMode">
        {{ isSignUp ? "Already have an account? Sign in" : "Need an account? Sign up" }}
      </button>
    </form>
  </div>
</template>

<style scoped>
.login {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--paper);
}

.login__card {
  display: flex;
  flex-direction: column;
  gap: 14px;
  width: 280px;
  padding: 28px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
}

.login__title {
  margin: 0;
  font-family: var(--font-display);
  font-size: 24px;
  font-weight: 500;
  color: var(--ink);
}

.login__subtitle {
  margin: -8px 0 4px;
  font-size: 13px;
  color: var(--ink-muted);
}

.login__field {
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 13px;
  color: var(--ink-muted);
}

.login__field input {
  padding: 9px 10px;
  font: inherit;
  color: var(--ink);
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  outline: none;
  transition: border-color 0.15s ease;
}

.login__field input:focus {
  border-color: var(--accent);
}

.login__error {
  margin: 0;
  font-size: 13px;
  color: var(--error);
}

.login__submit {
  margin-top: 4px;
  padding: 10px;
  font: inherit;
  font-weight: 500;
  color: var(--paper);
  background: var(--accent);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: opacity 0.15s ease;
}

.login__submit:hover:not(:disabled) {
  opacity: 0.88;
}

.login__submit:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.login__toggle {
  padding: 0;
  font-size: 12px;
  color: var(--ink-muted);
  background: transparent;
  border: none;
  cursor: pointer;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.login__toggle:hover {
  color: var(--accent);
}
</style>
