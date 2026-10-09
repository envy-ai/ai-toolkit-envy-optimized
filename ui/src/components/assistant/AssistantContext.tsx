'use client';

import {
  createContext,
  useContext,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type RefObject,
  type SetStateAction,
} from 'react';
import { MdSmartToy } from 'react-icons/md';

type AssistantState = {
  open: boolean;
  setOpen: Dispatch<SetStateAction<boolean>>;
  busy: boolean;
  setBusy: Dispatch<SetStateAction<boolean>>;
  trigger: RefObject<HTMLButtonElement | null>;
};

const AssistantContext = createContext<AssistantState | null>(null);

export function AssistantContextProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);

  return (
    <AssistantContext.Provider value={{ open, setOpen, busy, setBusy, trigger }}>{children}</AssistantContext.Provider>
  );
}

export function useAssistant() {
  const assistant = useContext(AssistantContext);
  if (!assistant) throw new Error('Assistant controls require AssistantContextProvider.');
  return assistant;
}

export function AssistantToggleButton() {
  const { open, setOpen, busy, trigger } = useAssistant();
  return (
    <button
      ref={trigger}
      type="button"
      title="AI assistant"
      aria-label={open ? 'Close AI assistant' : 'Open AI assistant'}
      aria-expanded={open}
      aria-controls="toolkit-assistant"
      onClick={() => setOpen(current => !current)}
      className="ml-2 flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-gray-300 hover:bg-gray-800 hover:text-white focus-visible:outline focus-visible:outline-blue-400"
    >
      <MdSmartToy aria-hidden="true" className={`h-5 w-5 ${busy ? 'animate-pulse text-blue-400' : ''}`} />
    </button>
  );
}
