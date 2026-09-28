import type { ModelStatus } from "./models";
import type { Account, Chat, ChatImage, ChatSettings } from "./types";

const API_ORIGIN = import.meta.env.VITE_API_ORIGIN ?? "http://localhost:8000";
const BASE_URL = `${API_ORIGIN}/api`;
const TOKEN_KEY = "reffusion_token";
const USERNAME_KEY = "reffusion_username";

export interface StoredAuth {
  token: string;
  username: string;
}

export function getStoredAuth(): StoredAuth | null {
  const token = localStorage.getItem(TOKEN_KEY);
  const username = localStorage.getItem(USERNAME_KEY);
  return token && username ? { token, username } : null;
}

export function clearStoredAuth(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USERNAME_KEY);
}

function currentUsername(): string {
  const auth = getStoredAuth();
  if (!auth) throw new Error("Not logged in.");
  return auth.username;
}

/** Flattens a DRF error body ({field: [msg, ...], ...}) into one string. */
async function describeError(response: Response): Promise<string> {
  const fallback = `Request failed (${response.status} ${response.statusText})`;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object") {
      const messages = Object.values(body as Record<string, unknown>).flat();
      if (messages.length) return messages.join(" ");
    }
  } catch {
    // Response wasn't JSON; fall through to the generic status message.
  }
  return fallback;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const auth = getStoredAuth();
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: {
      Accept: "application/json",
      ...(options?.body ? { "Content-Type": "application/json" } : {}),
      ...(auth ? { Authorization: `Token ${auth.token}` } : {}),
    },
    ...options,
  });

  if (response.status === 401) {
    clearStoredAuth();
  }

  if (!response.ok) {
    throw new Error(await describeError(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

/** Generated images come back as media-relative paths; point them at Django's origin. */
export function mediaUrl(path: string): string {
  return path.startsWith("http") ? path : `${API_ORIGIN}${path}`;
}

async function authenticate(path: string, username: string, password: string): Promise<void> {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    throw new Error(await describeError(response));
  }

  const { token } = (await response.json()) as { token: string };
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USERNAME_KEY, username);
}

export function login(username: string, password: string): Promise<void> {
  return authenticate("/login/", username, password);
}

export function signup(username: string, password: string): Promise<void> {
  return authenticate("/signup/", username, password);
}

export async function logout(): Promise<void> {
  try {
    // Best-effort: revoke the token server-side so it can't be reused.
    // Still clear local state even if this fails (e.g. it already expired
    // server-side, or we're offline) - the user asked to be logged out.
    await request<void>("/logout/", { method: "POST" });
  } catch {
    // ignore
  } finally {
    clearStoredAuth();
  }
}

export function fetchAccount(): Promise<Account> {
  return request<Account>(`/accounts/${currentUsername()}/`);
}

export function createChat(settings: ChatSettings): Promise<Chat> {
  return request<Chat>(`/accounts/${currentUsername()}/new_chat/`, {
    method: "POST",
    body: JSON.stringify(settings),
  });
}

export function deleteChat(chatId: string): Promise<void> {
  return request<void>(`/accounts/${currentUsername()}/chat/${chatId}/`, { method: "DELETE" });
}

export function postPrompt(chatId: string, text: string): Promise<ChatImage> {
  return request<ChatImage>(`/accounts/${currentUsername()}/chat/${chatId}/`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export function fetchModelStatuses(): Promise<ModelStatus[]> {
  return request<ModelStatus[]>("/models/");
}

export function warmModel(modelKey: string): Promise<void> {
  return request<void>(`/models/${modelKey}/warm/`, { method: "POST" });
}
