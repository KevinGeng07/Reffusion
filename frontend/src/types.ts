export interface ChatImage {
  log_id: number;
  text: string;
  image: string;
  comparison_image: string | null;
}

export interface Chat {
  chat_id: string;
  name: string;
  use_conditioning: boolean;
  conditioning_strength: number;
  num_inference_steps: number;
  model_key: string;
  chat_log: ChatImage[];
}

export type ChatSettings = Pick<
  Chat,
  | "name"
  | "use_conditioning"
  | "conditioning_strength"
  | "num_inference_steps"
  | "model_key"
>;

export interface Account {
  name: string;
  user_id: string;
  chat: Chat[];
}
