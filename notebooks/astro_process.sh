#!/data/data/com.termux/files/usr/bin/bash

# --- Paramètres par défaut ---
METHOD="mean" ; ALGO="-q 3" ; OUT="final_result.png"
GAMMA=2.2 ; CONTRAST=5 ; SAT=130 ; SHARP=0.8 ; POLLUTION=false
BINNING=""
BLACK_POINT="25%"   

usage() {
    echo "Usage: $0 -l 'LIGHTS*.CR2' [-d 'DARKS*.CR2'] [-f 'FLATS*.CR2'] [-b] [-p] [-o m31]"
    exit 1
}

while getopts "l:d:f:m:o:g:c:s:n:p:bh" opt; do
    case $opt in
        l) L_SRC=$OPTARG ;;
        d) D_SRC=$OPTARG ;;
        f) F_SRC=$OPTARG ;;
        m) METHOD=$OPTARG ;;
        o) OUT=$OPTARG ;;
        g) GAMMA=$OPTARG ;;
        c) CONTRAST=$OPTARG ;;
        s) SAT=$OPTARG ;;
        n) SHARP=$OPTARG ;;
        p) BLACK_POINT=$OPTARG ;;
        #p) POLLUTION=true ;;
        b) BINNING="-h" ;;
        h) usage ;;
    esac
done

if [ -z "$L_SRC" ]; then usage; fi

# --- Correction automatique de l'extension ---
if [[ "$OUT" != *.png ]]; then
    BASENAME_OUT="$OUT"
    OUT="${OUT}.png"
else
    BASENAME_OUT="${OUT%.*}"
fi

rm -rf .temp_calib .temp_aligned
mkdir -p .temp_calib .temp_aligned

# --- Creation des Master ---
make_master() {
    local type=$1; local files=($2); local master=$3
    local total=${#files[@]}
    if [ ! -f "$master" ] && [ $total -gt 0 ]; then
        echo "--- Création Master $type ($total images) ---"
        count=0
        for f in "${files[@]}"; do
            [ -e "$f" ] || continue
            ((count++))
            echo -ne "   Progression : $count/$total\r"
            source_dir=$(dirname "$f")
            name_no_ext=$(basename "${f%.*}")
            dcraw_emu -4 -T $BINNING "$f" > /dev/null 2>&1
            gen=$(ls "$source_dir/${name_no_ext}"*".tiff" 2>/dev/null | head -n 1)
            [ -f "$gen" ] && mv "$gen" ".temp_calib/${name_no_ext}.tiff"
        done
        echo -e "\n   Empilement du Master..."
        magick -limit memory 1024MiB .temp_calib/*.tiff -evaluate-sequence mean -auto-level "$master"
        rm .temp_calib/*.tiff
    fi
}

# 1. Masters
make_master "Dark" "$D_SRC" "dark_maitre.tiff"
make_master "Flat" "$F_SRC" "flat_maitre.tiff"

# 2. Calibration
echo "--- Calibration des Lights ---"
light_files=($L_SRC)
total_lights=${#light_files[@]}
count=0
for f in "${light_files[@]}"; do
    [ -e "$f" ] || continue
    ((count++))
    echo -ne "   Traitement : $count/$total_lights\r"
    source_dir=$(dirname "$f")
    name_no_ext=$(basename "${f%.*}")
    dcraw_emu -4 -T $ALGO $BINNING "$f" > /dev/null 2>&1
    gen=$(ls "$source_dir/${name_no_ext}"*".tiff" 2>/dev/null | head -n 1)
    if [ -f "$gen" ]; then
        if [ -f "dark_maitre.tiff" ] && [ -f "flat_maitre.tiff" ]; then
               magick -limit memory 1024MiB "$gen" dark_maitre.tiff -compose subtract -composite \
                    flat_maitre.tiff -compose divide -composite ".temp_calib/${name_no_ext}.tiff"
            rm "$gen"
        else
            mv "$gen" ".temp_calib/${name_no_ext}.tiff"
        fi
    fi
done
echo -e "\n"

# 3. Alignement
echo "--- Alignement ---"
FILES=(.temp_calib/*.tiff)
total_align=${#FILES[@]}
REF="${FILES[0]}"
count=0
for f in "${FILES[@]}"; do
    ((count++))
    dest=".temp_aligned/$(basename "$f")"
    echo -ne "   Alignement : $count/$total_align\r"
    [ "$f" == "$REF" ] && cp "$f" "$dest" || python align_astro.py "$REF" "$f" "$dest"
done
echo -e "\n"

# 4. Stacking
echo "--- Empilement ($METHOD) ---"
if [ "$METHOD" == "add" ] && op="sum" || op="mean"; then
      magick -limit memory 1024MiB .temp_aligned/*.tiff -evaluate-sequence $op -depth 16 "PNG48:.stack_result.png"
fi

# 5. Anti-Pollution

if [ "$POLLUTION" = true ]; then
    echo "--- Retrait pollution lumineuse ---"
    magick -limit memory 1024MiB "stack_result.png" -gaussian-blur 100x100 -depth 16 PNG48:.temp_bg.tiff
    magick -limit memory 1024MiB "stack_result.png" .temp_bg.tiff -compose subtract -composite -depth 16 "PNG48:.stack_result.png"
    rm .temp_bg.tiff
fi

# 6. Post-Process 

# --- Étape finale : Développement du rendu ---
echo "--- Améliorations (Point Noir : $BLACK_POINT) ---"

magick -limit memory 1024MiB ".stack_result.png" \
    -separate -normalize -combine \
    -gamma "$GAMMA" \
    -modulate "100,$SAT" \
    -level "$BLACK_POINT",100%,1.2 \
    -unsharp "0x1+$SHARP+0.008" \
    -sigmoidal-contrast "${CONTRAST}x50%" \
    -median 3 \
    -depth 16 \
    PNG48:"${BASENAME_OUT}.png"

# Nettoyage
rm -rf .temp_calib .temp_aligned 
#rm -rf .temp_calib .temp_aligned dark_maitre.tiff flat_maitre.tiff
echo "------------------------------------------"
echo "TERMINÉ !"
echo "Résultat 16-bit (PNG) : $OUT"


