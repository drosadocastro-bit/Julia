# Cibuco_Boriken — GitHub Upload + Colab CFAR Run
# Team: Cibuco_Boriken 🇵🇷🌿
# Date: March 12, 2026
# Goal: Push birdclef module to GitHub → run CFAR k-sweep on Colab GPU
#
# Pass this to Opus with:
# "Work through this TODO top to bottom.
#  Generate any files marked with [OPUS BUILDS THIS]"

# ══════════════════════════════════════════════════════
# PHASE 1 — PREP BEFORE GITHUB (local, ~10 min)
# ══════════════════════════════════════════════════════

PHASE_1 = [

    {
        "id": "1-01",
        "task": "Create .gitignore",
        "who": "OPUS BUILDS THIS",
        "details": """
            Create .gitignore in project root.
            Must exclude:
              data/               ← 16GB competition data, never push
              birdclef/models/    ← trained model weights
              birdclef/output/    ← submission csvs
              __pycache__/
              *.pyc
              *.pt                ← PyTorch model files
              *.pth
              .env
              *.zip
              k_sweep_figure.png  ← generated figures
              birdclef_sanity_*/  ← temp test dirs
            Must INCLUDE:
              birdclef/           ← all source code
              tests/              ← all tests
              README.md
              requirements.txt
              birdclef_colab_cfar.ipynb
        """,
    },

    {
        "id": "1-02",
        "task": "Create requirements.txt",
        "who": "OPUS BUILDS THIS",
        "details": """
            Generate requirements.txt with pinned versions:
              torch
              torchaudio
              torchvision
              librosa
              soundfile
              numpy
              pandas
              scikit-learn
              transformers    ← for Perch backbone
              kaggle          ← for data download
              jupyter         ← for notebook
            Use versions currently installed in the project.
            On Windows PowerShell, use:
            pip freeze | Select-String "torch|torchaudio|torchvision|librosa|soundfile|numpy|pandas|scikit-learn|transformers|kaggle|jupyter"
        """,
    },

    {
        "id": "1-03",
        "task": "Create README.md",
        "who": "OPUS BUILDS THIS",
        "details": """
            Professional README for cibuco-boriken repo.
            
            Sections:
            
            # Cibuco_Boriken — BirdCLEF+ 2026
            Team name meaning: Cibuco (river, Manatí PR) + Boriken (Taíno name for Puerto Rico)
            
            ## About
            BirdCLEF+ 2026 competition entry.
            Brazilian Pantanal biodiversity monitoring.
            Novel contribution: CFAR-inspired adaptive thresholding
            borrowed from radar signal processing applied to
            bioacoustic species detection.
            
            ## Architecture
            - ARIA audio pipeline adapted for 234-species multilabel classification
            - 50% overlap windowing for soundscape inference
            - Weighted BCE loss for rare species
            - CFAR adaptive thresholding (our novel contribution)
            - Backbones: SmallCNN, MobileNetV2, EfficientNet-B0, ResNet-18
            
            ## Quick Start
            pip install -r requirements.txt
            kaggle competitions download -c birdclef-2026
            python -m birdclef.train --backbone small --epochs 10
            python -m birdclef.evaluate_thresholds --k-sweep 1.0 1.5 2.0 2.5 3.0
            
            ## Colab
            Open birdclef_colab_cfar.ipynb in Google Colab for GPU training.
            
            ## Working Note
            "CFAR-Inspired Adaptive Thresholding for Bioacoustic 
            Species Detection in the Brazilian Pantanal"
            Submitted to CLEF 2026.
            
            ## Team
            Danny (Cibuco_Boriken) — FAA Air Traffic Systems Specialist,
            Bayamón Puerto Rico. 20+ years signal processing experience.
            
            ## License
            MIT
        """,
    },

    {
        "id": "1-04",
        "task": "Verify .gitignore works — check data/ is excluded",
        "who": "DANNY runs this",
        "command": "git status --short | Select-Object -First 30",
        "check": "data/ folder should NOT appear in output",
    },

]


# ══════════════════════════════════════════════════════
# PHASE 2 — GITHUB PUSH (~5 min)
# ══════════════════════════════════════════════════════

PHASE_2 = [

    {
        "id": "2-01",
        "task": "Initialize git repo",
        "who": "DANNY runs this",
        "commands": [
            "cd your_birdclef_project_folder",
            "git init",
            "git add .",
            'git commit -m "Cibuco_Boriken — BirdCLEF+ 2026 initial scaffold"',
        ],
    },

    {
        "id": "2-02",
        "task": "Create GitHub repo and push",
        "who": "DANNY runs this",
        "commands": [
            "gh repo create cibuco-boriken --public --push --source=.",
        ],
        "alternative": """
            If gh CLI not working:
            1. Go to github.com/new
            2. Name: cibuco-boriken
            3. Public ✅
            4. NO readme (we have one already)
            5. Copy the remote URL
            6. git remote add origin <url>
            7. git push -u origin main
        """,
    },

    {
        "id": "2-03",
        "task": "Verify repo live",
        "who": "DANNY checks this",
        "check": "Visit github.com/drosadocastro-bit/cibuco-boriken",
        "expected": [
            "All birdclef/ files visible ✅",
            "data/ folder NOT visible ✅",
            "README.md renders on homepage ✅",
            "birdclef_colab_cfar.ipynb visible ✅",
        ],
    },

]


# ══════════════════════════════════════════════════════
# PHASE 3 — COLAB SETUP (~5 min)
# ══════════════════════════════════════════════════════

PHASE_3 = [

    {
        "id": "3-01",
        "task": "Open Colab and set runtime to GPU",
        "who": "DANNY does this",
        "steps": [
            "Go to colab.research.google.com",
            "Upload birdclef_colab_cfar.ipynb OR open from Drive",
            "Runtime → Change runtime type → T4 GPU",
            "Connect",
        ],
        "note": "T4 GPU = ~8 compute units/hour. 500 samples run ≈ 0.5 units. Safe. 🎯",
    },

    {
        "id": "3-02",
        "task": "Cell 1 — Clone repo + install deps",
        "who": "OPUS BUILDS THIS CELL",
        "code": """
            !git clone https://github.com/drosadocastro-bit/cibuco-boriken
            %cd cibuco-boriken
            !pip install -q -r requirements.txt
            print("Setup complete ✅")
        """,
    },

    {
        "id": "3-03",
        "task": "Cell 2 — Kaggle credentials + data download",
        "who": "OPUS BUILDS THIS CELL",
        "code": """
            # Upload kaggle.json when prompted
            from google.colab import files
            import os

            uploaded = files.upload()  # upload kaggle.json here
            os.makedirs('/root/.kaggle', exist_ok=True)
            !cp kaggle.json /root/.kaggle/
            !chmod 600 /root/.kaggle/kaggle.json

            # Download competition data
            !kaggle competitions download -c birdclef-2026
            !unzip -q birdclef-2026.zip -d data/birdclef-2026
            print("Data ready ✅")
        """,
        "note": "kaggle.json is at ~/.kaggle/kaggle.json locally. Download it from kaggle.com/settings",
    },

    {
        "id": "3-04",
        "task": "Cell 3 — Mount Drive (save model after training)",
        "who": "OPUS BUILDS THIS CELL",
        "code": """
            from google.colab import drive
            drive.mount('/content/drive')
            SAVE_DIR = '/content/drive/MyDrive/cibuco_boriken/'
            import os
            os.makedirs(SAVE_DIR, exist_ok=True)
            print(f"Drive mounted. Saving to: {SAVE_DIR}")
        """,
    },

]


# ══════════════════════════════════════════════════════
# PHASE 4 — COLAB TRAINING + CFAR RUN (~20-30 min)
# ══════════════════════════════════════════════════════

PHASE_4 = [

    {
        "id": "4-01",
        "task": "Cell 4 — Training run (500 samples, 10 epochs)",
        "who": "OPUS BUILDS THIS CELL",
        "command": """
            !python -m birdclef.train \\
              --backbone small \\
              --epochs 10 \\
              --max-samples 500 \\
              --include-soundscapes
        """,
        "expected_time": "~5-8 minutes on T4 GPU",
        "watch_for": [
            "Loss dropping each epoch ✅",
            "No NaN loss ✅",
            "Model saved ✅",
        ],
    },

    {
        "id": "4-02",
        "task": "Cell 5 — CFAR k-sweep on real trained model",
        "who": "OPUS BUILDS THIS CELL",
        "command": """
            !python -m birdclef.evaluate_thresholds \\
              --backbone small \\
              --max-samples 500 \\
              --include-soundscapes \\
              --k-sweep 1.0 1.5 2.0 2.5 3.0
        """,
        "this_is_the_moment": "Real CFAR results on calibrated model 🎯",
        "watch_for": [
            "F1 rare species: CFAR > fixed at some k value",
            "FPR: CFAR ≤ fixed at optimal k",
            "Visible knee in the k curve",
            "T_mean >> 0.05 (model producing real activations)",
        ],
    },

    {
        "id": "4-03",
        "task": "Cell 6 — Display and save figure",
        "who": "OPUS BUILDS THIS CELL",
        "code": """
            from IPython.display import Image, display
            import shutil

            display(Image('k_sweep_figure.png'))

            # Save to Drive
            shutil.copy('k_sweep_figure.png',
                       SAVE_DIR + 'k_sweep_500samples.png')
            shutil.copy('birdclef/models/birdclef_model.pt',
                       SAVE_DIR + 'birdclef_model_500samples.pt')
            print("Saved to Drive ✅")
        """,
    },

    {
        "id": "4-04",
        "task": "Cell 7 — Print paper-ready results table",
        "who": "OPUS BUILDS THIS CELL",
        "code": """
            # Format results as markdown table for working note
            import json
            from pathlib import Path

            rows = json.loads(Path('k_sweep_results.json').read_text(encoding='utf-8'))
            print("## Table 1: CFAR k-Sensitivity Results (500 samples)")
            print("| k | F1 Fixed | F1 CFAR | FPR Fixed | FPR CFAR | T_mean |")
            print("|---|----------|---------|-----------|----------|--------|")
            for r in rows:
                print(f"| {r['k']:.1f} | {r['f1_fixed']:.4f} | {r['f1_cfar']:.4f} | {r['fpr_fixed']:.4f} | {r['fpr_cfar']:.4f} | {r['threshold_mean']:.4f} |")
        """,
    },

]


# ══════════════════════════════════════════════════════
# SUCCESS CRITERIA — How we know it worked
# ══════════════════════════════════════════════════════

SUCCESS = {
    "GitHub": [
        "github.com/drosadocastro-bit/cibuco-boriken is live ✅",
        "README renders correctly ✅",
        "No data/ folder in repo ✅",
    ],
    "Colab": [
        "Training completes without errors ✅",
        "val_loss < 0.5 after 10 epochs ✅",
        "T_mean > 0.05 (real activations, not floor-clamped) ✅",
    ],
    "CFAR": [
        "k sweep shows visible knee in curve ✅",
        "At least one k value: CFAR F1_rare > Fixed F1_rare ✅",
        "k_sweep_figure.png saved to Drive ✅",
        "Results table ready for paper Section 4 ✅",
    ],
    "Paper": [
        "Section 3.1 method: DONE ✅",
        "Section 3.2 k-sensitivity: DONE ✅",
        "Section 4 results: PENDING (this run) ⏳",
        "Figure 1: PENDING (this run) ⏳",
    ],
}

COMPUTE_BUDGET = {
    "Available": "83 units",
    "This run (500 samples)": "~0.5 units",
    "Remaining after": "~82.5 units",
    "Full dataset run (when ready)": "~8-10 units",
    "Safety reserve": "keep 20 units for final paper run",
}
