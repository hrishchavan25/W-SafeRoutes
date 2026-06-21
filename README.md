# W-SafeRoutes

Women safety route planning app with an Expo/React Native frontend and deployable backend services.

## Local development

Install frontend dependencies:

```bash
npm install
```

Run the Expo app:

```bash
npm start
```

Run the ML backend:

```bash
pip install -r requirements.txt
uvicorn colorpredict3:app --host 0.0.0.0 --port 8100
```

Run the auth backend:

```bash
npm run server
```

## Deployment

The repo includes:

- `.github/workflows/pages.yml` for GitHub Pages web deployment.
- `render.yaml` and `Procfile` for deploying the Python ML API and Node auth API on Render.

Production frontend URLs are configured with:

- `EXPO_PUBLIC_ML_API_URL`
- `EXPO_PUBLIC_AUTH_API_URL`
- `EXPO_PUBLIC_SOS_API_URL`

The auth API uses `MONGODB_URI` when deployed. Keep local/private user data in `users.json`; it is intentionally ignored by git.
