/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_WAREHOUSE_WS_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
