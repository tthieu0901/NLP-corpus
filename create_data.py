import os
import re
import xml.etree.ElementTree as ET
import xml.dom.minidom
from collections import defaultdict

def format_id(num):
    return f"{num:03}"

def extract_metadata(content, field):
    pattern = fr"{field}:\s*(.*?)\s*$"
    match = re.search(pattern, content, re.MULTILINE)
    return match.group(1).strip() if match else ""

def create_data_hieu(folder_path, output_path="output.xml"):
    count_sentence = 0

    # Gom nhóm file theo chapter (lấy từ tên file)
    chapter_groups = defaultdict(list)

    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".txt"):
            continue

        name_parts = filename[:-4].split("_")
        if len(name_parts) != 4:
            continue

        chapter, volumn, section, page = name_parts
        page = int(page)
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chapter_groups[chapter].append({
            "filename": filename,
            "filepath": filepath,
            "content": content,
            "chapter": chapter,
            "volumn": volumn,
            "section": section,
            "page": page
        })

    # Root
    root = ET.Element("ROOT")

    for chapter_index, (chapter, files) in enumerate(chapter_groups.items(), start=1):
        file_elem = ET.SubElement(root, "FILE", ID="HCS_007")

        # META
        meta = ET.SubElement(file_elem, "META")
        ET.SubElement(meta, "TITLE").text = chapter
        ET.SubElement(meta, "VOLUMN").text = "Minh sử"
        ET.SubElement(meta, "AUTHOR").text = ""
        ET.SubElement(meta, "PERIOD").text = "Nhà Thanh"
        ET.SubElement(meta, "LANGUAGE").text = "Hán"

        # Lấy source từ file đầu tiên (nếu có)
        source = extract_metadata(files[0]["content"], "URL")
        ET.SubElement(meta, "SOURCE").text = source

        # Gom section
        section_map = defaultdict(list)
        for f in files:
            section_map[f["section"]].append(f)

        for sect_index, (section, section_files) in enumerate(section_map.items(), start=1):
            sect_id = f"HCS_007.{format_id(sect_index)}"
            source = extract_metadata(section_files[0]["content"], "URL")
            sect_elem = ET.SubElement(file_elem, "SECT", ID=sect_id, NAME=section, SOURCE=source)

            # Sort theo page
            section_files.sort(key=lambda x: x["page"])

            for page_index, file in enumerate(section_files, start=1):
                page_id = f"{sect_id}.{format_id(page_index)}"
                page_elem = ET.SubElement(sect_elem, "PAGE", ID=page_id)

                # Lấy phần thân sau ======
                parts = re.split(r"=+\s*", file["content"])
                if len(parts) < 2:
                    continue

                paragraph = parts[1].strip()
                sentences = re.findall(r".+?。", paragraph)

                for stc_index, sentence in enumerate(sentences, start=1):
                    stc_id = f"{page_id}.{format_id(stc_index)}"
                    ET.SubElement(page_elem, "STC", ID=stc_id).text = sentence.strip()
                    count_sentence += 1

    # Print number of sectences
    print(f"Số lượng câu: {count_sentence}")

    # Pretty print
    xml_str = ET.tostring(root, encoding="utf-8")
    parsed = xml.dom.minidom.parseString(xml_str)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(parsed.toprettyxml(indent="  "))

# Main
if __name__ == "__main__":
    folder_path_hieu = "ming_history_chapters_new"
    output_file_hieu = "xml_data/hieu_data.xml"
    create_data_hieu(folder_path_hieu, output_file_hieu)

    # folder_path_phong = "test_input_data_phong"  
    # output_file_phong = "xml_data/phong_data.xml"


