import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        recommend: resolve(__dirname, 'recommend.html'),
        search: resolve(__dirname, 'search.html')
      }
    }
  }
})
