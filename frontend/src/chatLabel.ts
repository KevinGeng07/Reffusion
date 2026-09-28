import type { Chat } from "./types";

export function chatLabel(chatId: string): string {
  return chatId.replace(/^chat-/, "Chat ");
}

export function chatDisplayName(chat: Pick<Chat, "chat_id" | "name">): string {
  return chat.name.trim() ? chat.name : chatLabel(chat.chat_id);
}
