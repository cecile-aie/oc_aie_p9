# p9dg/utils/class_mappings.py

# --- Dictionnaires de classes et couleurs ---
class_labels = {
    "ADI":  "Tissu adipeux",
    "BACK": "Arrière-plan (fond sans tissu)",
    "DEB":  "Débris cellulaires / nécrose",
    "LYM":  "Lymphocytes (infiltrat immunitaire)",
    "MUC":  "Mucus (sécrétions)",
    "MUS":  "Tissu musculaire lisse",
    "NORM": "Muqueuse colique normale",
    "STR":  "Stroma associé au cancer",
    "TUM":  "Épithélium adénocarcinomateux (tumeur)"
}

class_colors = {
    "ADI":  "#F2CEEE",
    "BACK": "#D859D0",
    "DEB":  "#A50026",
    "LYM":  "#4575B4",
    "MUC":  "#74ADD1",
    "MUS":  "#D73027",
    "NORM": "#D4DF38",
    "STR":  "#762A83",
    "TUM":  "#000000"
}

def make_idx_mappings(class_to_idx):
    """Construit les mappings inverses à partir du Dataset."""
    idx_to_name = {i: class_labels[c] for c, i in class_to_idx.items()}
    idx_to_color = {i: class_colors[c] for c, i in class_to_idx.items()}
    name_to_color = {class_labels[c]: class_colors[c] for c in class_labels}
    return idx_to_name, idx_to_color, name_to_color
