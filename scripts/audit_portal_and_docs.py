import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

course_dir = "/Users/prateekpaul/Library/CloudStorage/GoogleDrive-prateekp@iiitd.ac.in/My Drive/IIITD/mygithub/Cheminformatics"
pages_dir = "/Users/prateekpaul/Library/CloudStorage/GoogleDrive-prateekp@iiitd.ac.in/My Drive/IIITD/mygithub/meprateekpaul.github.io"

print("="*80)
print("AUDITOR 3 VERIFICATION & COMPLIANCE SUITE")
print("="*80)

# 1. Check HTML portal files
portal_files = [
    os.path.join(pages_dir, "cheminformatics.html"),
    os.path.join(course_dir, "index.html")
]

categories_expected = [
    "all", "foundations", "representations", "descriptors",
    "searching", "qsar_ml", "structure_based", "capstone"
]

all_passed = True

for pfile in portal_files:
    print(f"\n--- Checking Portal File: {os.path.basename(pfile)} ---")
    if not os.path.exists(pfile):
        print(f"FAILED: File does not exist: {pfile}")
        all_passed = False
        continue
    
    with open(pfile, "r", encoding="utf-8") as f:
        content = f.read()
    
    print(f"File size: {len(content):,} bytes")
    
    # 1.1 Check Title and Meta
    title_match = re.search(r"<title>(.*?)</title>", content)
    print(f"Title: {title_match.group(1) if title_match else 'MISSING'}")
    
    # 1.2 Check Tailwind CSS and Google Fonts
    has_tailwind = "cdn.tailwindcss.com" in content
    has_fonts = "fonts.googleapis.com" in content
    has_fa = "fontawesome" in content or "font-awesome" in content
    print(f"Tailwind CSS CDN: {'YES' if has_tailwind else 'NO'}")
    print(f"Google Fonts: {'YES' if has_fonts else 'NO'}")
    print(f"Font Awesome: {'YES' if has_fa else 'NO'}")
    
    # 1.3 Check Filter Buttons
    buttons = re.findall(r'<button[^>]*class="[^"]*filter-btn[^"]*"[^>]*data-filter="([^"]+)"[^>]*>(.*?)</button>', content)
    print(f"Filter buttons found: {len(buttons)}")
    btn_filters = [b[0] for b in buttons]
    for b_filter, b_text in buttons:
        print(f"  - Filter: {b_filter:<16} Text: {b_text.strip()}")
    
    missing_cats = [c for c in categories_expected if c not in btn_filters]
    if missing_cats:
        print(f"FAILED: Missing filter buttons for categories: {missing_cats}")
        all_passed = False
    else:
        print("PASSED: All 8 interactive category filters are present.")
        
    # 1.4 Check Notebook Cards
    # Split content by <!-- nbXX --> comments
    card_sections = re.findall(r'<!-- (nb\d+) -->(.*?)(?=<!-- nb\d+ -->|</div>\s*</section>)', content, re.DOTALL)
    print(f"Notebook card sections found: {len(card_sections)}")
    if len(card_sections) != 22:
        print(f"FAILED: Expected 22 notebook cards, found {len(card_sections)}")
        all_passed = False
    else:
        print("PASSED: Exactly 22 notebook cards found.")
        
    for nb_id, card_body in card_sections:
        # Check category
        cat_match = re.search(r'data-category="([^"]+)"', card_body)
        cat_data = cat_match.group(1) if cat_match else "MISSING"
        
        # Check title
        h3 = re.search(r'<h3[^>]*>(.*?)</h3>', card_body)
        title = h3.group(1) if h3 else "MISSING"
        
        # Check difficulty pill
        diff = re.search(r'>(Beginner|Intermediate|Advanced)<', card_body)
        diff_val = diff.group(1) if diff else "MISSING"
        
        # Check GitHub link
        gh = re.search(r'href="(https://github\.com/meprateekpaul/Cheminformatics/blob/main/notebooks/[^"]+)"', card_body)
        gh_url = gh.group(1) if gh else "MISSING"
        
        # Check Colab link
        colab = re.search(r'href="(https://colab\.research\.google\.com/github/meprateekpaul/Cheminformatics/blob/main/notebooks/[^"]+)"', card_body)
        colab_url = colab.group(1) if colab else "MISSING"
        
        # Check topics
        topics = re.findall(r'<span class="bg-slate-100[^"]*">([^<]+)</span>', card_body)
        
        if diff_val == "MISSING" or gh_url == "MISSING" or colab_url == "MISSING" or not topics:
            print(f"FAILED on {nb_id}: Diff={diff_val}, GH={bool(gh)}, Colab={bool(colab)}, Topics={len(topics)}")
            all_passed = False
        else:
            print(f"  [{nb_id}] {cat_data:<15} | Diff: {diff_val:<12} | Topics: {len(topics)} | GH: OK | Colab: OK | {title}")

    # 1.5 Check JavaScript Filter Function
    has_js_filter = "function filterNotebooks" in content and "data-category" in content
    print(f"JavaScript Filter function: {'VALID & INTEGRATED' if has_js_filter else 'INVALID'}")
    if not has_js_filter:
        all_passed = False

    # 1.6 Check Navigation Links
    has_nav_portfolio = 'href="index.html"' in content
    has_nav_pcb = 'href="programmingcb.html"' in content
    has_nav_repo = 'href="https://github.com/meprateekpaul/Cheminformatics"' in content
    has_nav_about = 'href="#about"' in content and 'id="about"' in content
    has_nav_chemlib = 'href="#chemlib"' in content and 'id="chemlib"' in content
    has_nav_curriculum = 'href="#curriculum"' in content and 'id="curriculum"' in content
    has_nav_datasets = 'href="#datasets"' in content and 'id="datasets"' in content
    has_nav_assessments = 'href="#assessments"' in content and 'id="assessments"' in content
    
    print("\nNavigation Integrity Check:")
    print(f"  - index.html (Portfolio): {'OK' if has_nav_portfolio else 'MISSING'}")
    print(f"  - programmingcb.html (PCB): {'OK' if has_nav_pcb else 'MISSING'}")
    print(f"  - GitHub Repo: {'OK' if has_nav_repo else 'MISSING'}")
    print(f"  - Section #about: {'OK' if has_nav_about else 'MISSING'}")
    print(f"  - Section #chemlib: {'OK' if has_nav_chemlib else 'MISSING'}")
    print(f"  - Section #curriculum: {'OK' if has_nav_curriculum else 'MISSING'}")
    print(f"  - Section #datasets: {'OK' if has_nav_datasets else 'MISSING'}")
    print(f"  - Section #assessments: {'OK' if has_nav_assessments else 'MISSING'}")

# 2. Check navigation link in meprateekpaul.github.io/index.html
print("\n" + "="*80)
print("AUDITING meprateekpaul.github.io/index.html NAVIGATION LINK")
print("="*80)
index_html = os.path.join(pages_dir, "index.html")
with open(index_html, "r", encoding="utf-8") as f:
    idx_content = f.read()

has_chem_link = 'href="cheminformatics.html"' in idx_content
print(f"Link to cheminformatics.html in portfolio navbar: {'FOUND' if has_chem_link else 'MISSING'}")
if not has_chem_link:
    all_passed = False

# 3. Check Documentation in Cheminformatics/docs/
print("\n" + "="*80)
print("AUDITING COURSE DOCUMENTATION (Cheminformatics/docs/)")
print("="*80)

docs_to_check = [
    "slides_index.md",
    "glossary.md",
    "CHANGELOG.md",
    "question_bank.md",
    "cheat_sheets/smiles_smarts.md",
    "cheat_sheets/rdkit_guide.md"
]

for doc in docs_to_check:
    doc_path = os.path.join(course_dir, "docs", doc)
    exists = os.path.exists(doc_path)
    size = os.path.getsize(doc_path) if exists else 0
    print(f"  - docs/{doc:<30}: {'EXISTS' if exists else 'MISSING'} ({size:,} bytes)")
    if not exists:
        all_passed = False

# 4. Check Slide Errata E1 to E6 in docs/slides_index.md and Cheminformatics.md
print("\n" + "="*80)
print("AUDITING SLIDE ERRATA CORRECTIONS (E1 through E6)")
print("="*80)
slides_idx_path = os.path.join(course_dir, "docs", "slides_index.md")
with open(slides_idx_path, "r", encoding="utf-8") as f:
    s_content = f.read()

for e_num in ["E1", "E2", "E3", "E4", "E5", "E6"]:
    present = f"**{e_num}**" in s_content or f"| {e_num} |" in s_content
    print(f"  - Erratum {e_num} documented in docs/slides_index.md: {'YES' if present else 'NO'}")
    if not present:
        all_passed = False

# 5. Check Repository Top-Level Files
print("\n" + "="*80)
print("AUDITING REPOSITORY TOP-LEVEL FILES")
print("="*80)

top_files = [
    "README.md",
    "LICENSE",
    "Makefile",
    "environment.yml",
    "requirements.txt",
    "pyproject.toml",
    ".github/workflows/ci.yml"
]

for tf in top_files:
    tf_path = os.path.join(course_dir, tf)
    exists = os.path.exists(tf_path)
    size = os.path.getsize(tf_path) if exists else 0
    print(f"  - {tf:<28}: {'EXISTS' if exists else 'MISSING'} ({size:,} bytes)")
    if not exists:
        all_passed = False

# Verify notebook files exist
print("\n" + "="*80)
print("AUDITING 22 NOTEBOOK FILES & SOLUTIONS")
print("="*80)
from update_portal import nb_data
nb_dir = os.path.join(course_dir, "notebooks")
sol_dir = os.path.join(course_dir, "solutions")
for d in nb_data:
    nb_f = os.path.join(nb_dir, d["filename"])
    sol_f = os.path.join(sol_dir, d["filename"])
    nb_exists = os.path.exists(nb_f)
    sol_exists = os.path.exists(sol_f)
    if not nb_exists or not sol_exists:
        print(f"FAILED: {d['filename']} - Notebook: {nb_exists}, Solution: {sol_exists}")
        all_passed = False
    else:
        print(f"  - {d['id']}: {d['filename']} [Notebook: OK, Solution: OK]")

print("\n" + "="*80)
print(f"OVERALL AUDITOR 3 STATUS: {'CERTIFIED / PASSED ALL CHECKS' if all_passed else 'FAILED CHECKS'}")
print("="*80)
