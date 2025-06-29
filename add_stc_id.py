import xml.etree.ElementTree as ET
import xml.dom.minidom
import re

# Step 1: Strip whitespace-only text nodes recursively
def strip_whitespace(elem):
    if elem.text is not None and elem.text.strip() == "":
        elem.text = None
    if elem.tail is not None and elem.tail.strip() == "":
        elem.tail = None
    for child in elem:
        strip_whitespace(child)

# Step 2: Parse XML and clean it
tree = ET.parse("xml_data_ner/phong_data_ner_raw.xml")
root = tree.getroot()
strip_whitespace(root)

# Step 3: Add STC IDs
for page in root.findall(".//PAGE"):
    page_id = page.get("ID")
    for idx, stc in enumerate(page.findall("STC"), start=1):
        stc.set("ID", f"{page_id}.{idx:03d}")

# Step 4: Convert to string with minidom
xml_bytes = ET.tostring(root, encoding="utf-8")
parsed = xml.dom.minidom.parseString(xml_bytes)
pretty_xml = parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

# Step 5: Collapse multiline <STC> into one line
def collapse_stc(match):
    open_tag = match.group(1).strip()
    content = match.group(2)
    collapsed = ' '.join(content.split())
    return f"<STC {open_tag}>{collapsed}</STC>"

pretty_xml = re.sub(
    r"<STC([^>]*)>(.*?)</STC>",
    collapse_stc,
    pretty_xml,
    flags=re.DOTALL
)

# Step 6: Save result
with open("xml_data_ner/phong_data_ner.xml", "w", encoding="utf-8") as f:
    f.write(pretty_xml)
