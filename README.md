 🧠✨ AI Content Detector

Welcome to AI Content Detector — a simple, friendly tool that helps you figure out if an image or video is AI-generated. Whether you're curious or cautious, this tool is designed especially for non-techy folks like parents, grandparents, or anyone who wants to know what's real online.

> 🤖 “Was this photo made by a computer?”  
> 📹 “Is this video real or fake?”  
>  
> Now you can find out — quickly and easily.

---

 🧩 What It Does

✅ Detects AI-generated images using a custom-trained model  
🎥 Detects deepfake videos using a pretrained video classifier  
🧼 Clean and simple web interface  
📱 Works on phones and computers  
💡 Gives a short explanation along with the result

---

 🛠️ How It Works

This tool has two brains:

- 🖼️ A custom-trained model for images — built using real and fake samples from Midjourney, DALL·E, and more.
- 🎞️ A plug-and-play smart model for videos — using research-grade tools like DeepFake Detection Challenge (DFDC) models.

Your media is sent securely to our detector, processed, and you get a clear result:  
“Looks Real” or “Possibly AI-Generated”, with a short reason why.

---

 💻 Tech Stack (for nerds 👓)

| Part       | Tech                        |
|------------|-----------------------------|
| Frontend   | Vite + React + Tailwind CSS |
| Backend    | FastAPI + PyTorch           |
| Video AI   | Pretrained DFDC model       |
| Hosting    | Vercel + Render             |

---

 📁 Folder Peek


ai-content-detector/
├── backend/      ← FastAPI app + ML models
├── frontend/     ← React UI for uploading and showing results
├── data/         ← Real and fake training images
├── docs/         ← Full technical spec
├── README.md     ← You are here!

````

---

 🚀 Get Started (for contributors)

 🧠 Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
````

 🎨 Frontend

```bash
cd frontend
npm install
npm run dev
```

Open your browser to `http://localhost:3000` and start detecting ✨


 📖 Want to Learn More?

See the full technical design in [`docs/SPEC.adoc`](docs/SPEC.adoc).

---

 📜 License

MIT – free to use, remix, and share.

---

Built with care for my family — and yours. 💛
Because the truth should be easy to spot.

