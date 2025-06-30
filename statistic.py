import xml.etree.ElementTree as ET

def count_xml_elements(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    count_file = 0
    count_sect = 0
    count_page = 0
    count_stc = 0

    for file_elem in root.findall("FILE"):
        count_file += 1

        for sect_elem in file_elem.findall("SECT"):
            count_sect += 1

            for page_elem in sect_elem.findall("PAGE"):
                count_page += 1

                stc_elems = page_elem.findall("STC")
                count_stc += len(stc_elems)

    print(f"XML: {xml_path}")
    print(f"Total FILE: {count_file}")
    print(f"Total SECT: {count_sect}")
    print(f"Total PAGE: {count_page}")
    print(f"Total STC: {count_stc}")
    print(f"\n")

count_xml_elements("xml_data\hieu_data.xml")
count_xml_elements("xml_data\phong_data.xml")