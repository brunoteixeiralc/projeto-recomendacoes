<p align="center">
  <h1 align="center">🛍️ Recommendation Project</h1>
</p>

<p align="center">
  <strong>Intelligent recommendation system that personalizes the e-commerce showcase in real-time without impacting interface performance.</strong>
</p>

<p align="center">
  <img alt="JavaScript" src="https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E">
  <img alt="TensorFlow.js" src="https://img.shields.io/badge/TensorFlow.js-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img alt="HTML5" src="https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white">
  <img alt="CSS3" src="https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white">
  <img alt="Jest" src="https://img.shields.io/badge/-jest-%23C21325?style=for-the-badge&logo=jest&logoColor=white">
  <img alt="Supabase" src="https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white">
</p>

<hr>

## 📖 About the Project
This experimental project is a Vanilla JS e-commerce with a modularized architecture (quasi-MVC: Models/Data, Views, Controllers, Services, and Workers). Its main highlight is the **on-the-fly construction and training of a Machine Learning model** using `TensorFlow.js`.

The model's objective is to run in a Web Worker to perform *One-Hot Encoding* on variables such as prices, categories, colors, and ages, aimed at "learning" the audience profile for each product to **provide personalized recommendations based on the intersection of what the user has previously purchased**.

## ✨ Features
- **Profile Selection**: Users simulate access with a profile (extracted from a users `.json`) to access the platform.
- **Purchase History Tracking**: Tracks what the user clicks and "buys", persisted in memory/session.
- **Modular Architecture**: Robust implementation of Controllers, independent Views, and event-guided communication.
- **State Isolation (OOP)**: The AI training Worker uses Object-Oriented Programming (`RecommendationEngine`) to avoid global scope leaks.
- **Artificial Intelligence**: Neural Network training happens silently in the background using Web Workers without blocking the UI (`src/workers/modelTrainingWorker.js`).
- **Vector Database (Supabase)**: Integration with Supabase using `pgvector` for persistence and similarity search of product embeddings.
- **Proactive Security**: Sensitive credential management via `src/config.js` (Git ignored) and secure communication with the Worker.

## 🚀 How to Run Locally

Ensure you have [Node.js](https://nodejs.org/) installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USER/project-recommendations.git
   ```

2. Enter the directory:
   ```bash
   cd ecommerce-recommendations-with-ml
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Supabase Configuration:
   - Create `src/config.js` based on the provided template to include your `supabaseUrl` and `supabaseKey`.
   - Run the provided SQL setup script in your Supabase dashboard.

5. Start the local server:
   ```bash
   npm start
   ```

The page will open automatically via Browser-Sync (usually on port `3000`), with live-reloading.

## 🧪 Automated Testing

The project includes unit tests focused on the reliability of mathematical functions essential for TensorFlow Feature Engineering, such as normalization (`math.js`) and product context mapping (`dataProcessor.js`). We also test the `WorkerController` communication.

To run the test suite with **Jest**:

```bash
npm run test
```

## 🗂 Directory Structure
```bash
.
├── data/                    # Structured JSONs serving as local "Database"
├── src/
│   ├── controller/          # UI/Service orchestration rules
│   ├── events/              # Pub/Sub logic and internal events
│   ├── service/             # Async fetch requests
│   ├── utils/               # Utilities (dataProcessor.js, math.js)
│   ├── view/                # DOM / HTML manipulation
│   ├── workers/             # Web Workers (RecommendationEngine class)
│   ├── config.js            # Sensitive configurations (Git ignored)
│   └── index.js             # Application Entrypoint
├── index.html               # Main HTML structure
├── style.css                # Global Styling
└── package.json            
```

## 🧠 TensorFlow.js Logic (Step-by-Step)

The system's intelligence resides in `modelTrainingWorker.js`, functioning as an isolated "brain" in the background.

### 1. Initialization and Structure
Processing occurs in a **Web Worker**, ensuring the main thread (UI) never freezes. Logic is encapsulated in the `RecommendationEngine` class, managing model state and feature weights (Price, Category, Color, and Age).

### 2. Vectorization and Sync
- **Embeddings:** Transforms products into 16-dimensional numerical vectors.
- **Synchronization:** After calculation, vectors are sent to **Supabase** via `upsert`. This allows AI knowledge to be persistent and searchable via SQL using Cosine Similarity (`pgvector`).

### 3. Logic Workflow
1.  **Engine Init:** Loads product catalog and user profiles.
2.  **Feature Engineering:**
    *   **Products:** Converted to vectors based on price, color, category, and buyer's average age.
    *   **Users:** Profiles created by averaging purchased product vectors combined with their gender.
3.  **Neural Network (Training):** The model learns the relationship between user vectors and purchase probability.
4.  **Recommendation Cycle:**
    *   **Returning Users:** The network uses history and gender for predictions.
    *   **New Users (Cold Start):** Uses provided age and gender for immediate personalized predictions.

### 4. Communication
The Worker reports real-time progress and accuracy logs via a secure bridge controlled by the `WorkerController`.
