# Task 3: Data Visualization & Storytelling

## 📋 Overview
This task fulfills **CodeAlpha Data Analytics Internship - Task 3 (Data Visualization)**.
It transforms raw entertainment media metadata (2,500+ streaming catalog titles across Netflix, Prime Video, Disney+, and Apple TV+) into an executive-ready visual intelligence dashboard and data stories.

---

## 🎨 Visualization Design Principles
- **Clarity & Cognitive Load**: High-contrast, colorblind-friendly palettes (`viridis`, `coolwarm`, `mako`, custom brand accents).
- **Multi-Panel Cohesion**: Integrated GridSpec layouts combining proportion, trend, distribution, and cross-tabulation.
- **Narrative Data Storytelling**: Contextual callouts, threshold markers, and direct visual annotations replacing vague legends.

---

## 🖼️ Gallery of Visual Assets
All high-resolution outputs are located in [`outputs/`](outputs/):

| Visual Asset | Plot Type | Strategic Question Answered |
|---|---|---|
| `01_executive_dashboard.png` | Multi-panel GridSpec Dashboard | Comprehensive platform market share, release trajectory, and quality scores. |
| `02_genre_popularity_distribution.png` | Violin + Quartile Plot | Which genres consistently achieve high critical and audience acclaim? |
| `03_ratings_vs_duration_bubble.png` | Multi-variable Bubble Scatter | What is the optimal movie runtime for viewer engagement and high ratings? |
| `04_content_release_timeline.png` | Dual-Axis Combo Chart | How did rapid volume expansion impact average rating stability over 20 years? |
| `05_content_type_donut.png` | Donut & Pareto Bar | What is the structural split between feature films and serialized episodic series? |

---

## 🚀 How to Run
```bash
# Execute from the repository root:
python Task3_Data_Visualization/data_visualizer.py
```

---

## 📖 Key Takeaways from the Data Story
1. **The Quality-Volume Equilibrium**: Annual streaming catalog additions exploded 450% from 2015 to 2021. Median ratings stabilized near 6.5, showing that rapid expansion was achieved without catastrophic quality erosion.
2. **The 95–115 Minute Sweet Spot**: Films within the 95–115 minute envelope maximize total votes and positive sentiment.
3. **Prestige Niche Anchors**: Documentaries and Animation demonstrate the highest median ratings (>7.2), serving as critical retention anchors despite lower volume compared to mainstream comedies and dramas.
