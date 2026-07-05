"""
Demonstrator pentru calculul matematic al void-urilor (Arie vs Pixeli).
Folosit pentru a demonstra vizual iluzia optica a procentelor.
Acum cu o interfata prietenoasa in terminal si Toate bug-urile de tip unpacking rezolvate!
"""

import sys
from pathlib import Path
from math import pi

import cv2
import numpy as np

# Ne asiguram ca gaseste modulele tale din folderul 'src'
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_loader import load_grayscale
from src.preprocessing import PreprocessingConfig, preprocess_image
from src.ball_detection import HoughCircleConfig, detect_solder_balls_with_diagnostics, crop_ball_roi
from src.void_segmentation import VoidSegmentationConfig, segment_voids
from src.metrics import ComponentFilterConfig, analyze_void_components


def main():
    print("=" * 70)
    print(" 🔬 DEMONSTRATOR ILUZIE OPTICA: Pixeli Bruti vs Procentaj BGA")
    print("=" * 70)
    print("💡 Sfat: Poti sa scrii manual calea, SAU poti sa tragi poza (drag-and-drop)")
    print("         cu mouse-ul direct in aceasta fereastra!\n")

    # Citim calea de la utilizator (intuitiv, de la tastatura)
    user_input = input("👉 Te rog introdu calea catre poza de analizat (RAW):\n> ").strip()

    # Curatam ghilimelele automate pe care le pune Windows cand dai drag-and-drop
    user_input = user_input.strip('"').strip("'")

    if not user_input:
        print("\n❌ Nu ai introdus nicio cale. Programul se va inchide.")
        input("Apasa Enter pentru iesire...")
        return

    image_path = Path(user_input)
    if not image_path.exists():
        print(f"\n❌ EROARE: Nu gasesc nicio imagine la locatia:\n{image_path}")
        print("Asigura-te ca ai scris corect sau ca fisierul exista.")
        input("\nApasa Enter pentru iesire...")
        return

    print(f"\n⚙️  Incarcam '{image_path.name}'...")
    try:
        raw_gray = load_grayscale(image_path)
    except Exception as e:
        print(f"❌ EROARE la citirea imaginii: {e}")
        input("\nApasa Enter pentru iesire...")
        return

    # Cream un canvas intunecat (mai usor de vazut textul)
    canvas = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2BGR)

    # Incarcam configurarile tale actuale
    prep_config = PreprocessingConfig()
    ball_config = HoughCircleConfig()
    void_config = VoidSegmentationConfig()
    filter_config = ComponentFilterConfig()

    print("🔍 Procesam imaginea (asta poate dura cateva secunde)...")
    clahe, denoised = preprocess_image(raw_gray, prep_config)

    # Detectam bilele (FIX: Extragem lista din obiectul BallDetectionResult)
    detection_result = detect_solder_balls_with_diagnostics(denoised, ball_config)
    balls = detection_result.balls

    if not balls:
        print("⚠️ Nu am gasit nicio bila de cositor in aceasta imagine.")
        input("\nApasa Enter pentru iesire...")
        return

    print(f"✅ Am gasit {len(balls)} pad-uri. Calculam matematica pura...")

    for ball in balls:
        # FIX: crop_ball_roi returneaza imaginea, limitele si centrul
        roi_image, roi_bounds, _ = crop_ball_roi(denoised, ball)
        if roi_image is None or roi_image.size == 0:
            continue

        local_center = (
            ball.center_x - roi_bounds.x_min,
            ball.center_y - roi_bounds.y_min,
        )

        # Taiem ROI-ul brut pentru segmentare
        raw_roi = raw_gray[roi_bounds.y_min:roi_bounds.y_max, roi_bounds.x_min:roi_bounds.x_max]

        # Segmentam fix cum face si pipeline-ul mare
        void_mask = segment_voids(raw_roi, local_center, ball.radius, void_config)

        # Calculam Aria matematica a pad-ului (Pi * r^2)
        ball_area = pi * (ball.radius ** 2)

        # Filtram si numaram pixelii finali confirmati
        metrics, clean_mask = analyze_void_components(void_mask, ball_area, filter_config, raw_roi)

        # ----- DESENAM DOVEZILE PENTRU DEMONSTRATIE -----

        # 1. Desenam limita calculata a ariei (Cerc Verde)
        cv2.circle(canvas, (ball.center_x, ball.center_y), ball.radius, (0, 150, 0), 1, cv2.LINE_AA)

        # 2. Desenam doar pixelii confirmati (Masca Rosie)
        y_min, y_max = roi_bounds.y_min, roi_bounds.y_max
        x_min, x_max = roi_bounds.x_min, roi_bounds.x_max

        roi_canvas = canvas[y_min:y_max, x_min:x_max]
        red_layer = np.zeros_like(roi_canvas)
        red_layer[clean_mask > 0] = (0, 0, 255) # Rosu pur

        mask_indices = clean_mask > 0
        roi_canvas[mask_indices] = cv2.addWeighted(roi_canvas, 0.2, red_layer, 0.8, 0)[mask_indices]

        # 3. Desenam textele explicative
        if metrics.total_void_area > 0:
            ratio = (metrics.total_void_area / ball_area) * 100

            # Construim cele 3 randuri
            text_lines = [
                f"Arie Pad : {int(ball_area)} px",
                f"Pixeli Rosii: {metrics.total_void_area} px",
                f"Matematica: {ratio:.1f}%"
            ]

            # Gasim un loc deasupra pad-ului
            y_text = ball.center_y - ball.radius - 35

            # Adaugam putin background negru pentru a face textul lizibil
            for i, line in enumerate(text_lines):
                text_y = y_text + (i * 12)

                # Text cu Outline Negru (pentru vizibilitate pe raze X)
                cv2.putText(canvas, line, (ball.center_x - 35, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)

                # Culoarea textului interior
                color = (255, 255, 255) if i < 2 else (0, 255, 255) # Galben pt rezultat
                cv2.putText(canvas, line, (ball.center_x - 35, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Salvam rezultatul demonstrativ
    output_path = image_path.parent / f"{image_path.stem}_DEMO_MATH.jpg"
    cv2.imwrite(str(output_path), canvas)

    print("\n" + "=" * 70)
    print(f"🖼️  GATA! Imaginea cu dovezile matematice a fost salvata aici:\n➡️  {output_path}")
    print("=" * 70)

    # Tinem consola deschisa pana la apasarea unei taste
    input("\nApasa tasta Enter pentru a inchide programul...")


if __name__ == "__main__":
    main()