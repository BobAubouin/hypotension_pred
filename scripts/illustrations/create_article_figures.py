import fitz  # PyMuPDF

pdfs = [
    "./output/test_full_operationnal_prc.pdf",
    "./output/test_no_hypo_operationnal_prc.pdf",
    "./output/test_no_inter_operationnal_prc.pdf",
    "./output/test_no_hypo_no_inter_operationnal_prc.pdf",
]


labels = ["(a) Entire patient record", "(b) Normotensive set", "(c) Intervention free set", "(d) Normotensive-intervention free set"]
docs = [fitz.open(p) for p in pdfs]
pages = [d[0] for d in docs]

max_w = max(p.rect.width for p in pages)
max_h = max(p.rect.height for p in pages)

label_space = 40
gap = 12

page_width = 2 * max_w + gap
page_height = 2 * (max_h + label_space) + gap + 50

out = fitz.open()
page = out.new_page(width=page_width, height=page_height)

positions = [
    (0, 0),              # top-left
    (max_w + gap, 0),    # top-right
    (0, max_h + label_space + gap),                                      # bottom-left
    (max_w + gap, max_h + label_space + gap),                            # bottom-right
]

for src_page, label, (x, y) in zip(pages, labels, positions):
    # PDF area (above label)
    pdf_rect = fitz.Rect(
        x,
        y + label_space,
        x + max_w,
        y + label_space + max_h,
    )
    page.show_pdf_page(pdf_rect, src_page.parent, src_page.number)

    # Label area (below PDF)
    y+= max_h + 4*gap

    print(f'x={x}, y={y}')
    label_rect = fitz.Rect(
        x,
        y,
        x + max_w,
        y + label_space,
    )
    page.insert_textbox(
        label_rect,
        label,
        fontsize=18,
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER,
    )

out.save("Figure_4.pdf")
out.close()


pdfs = [
    "./output/chu_test_full_operationnal_prc.pdf",
    "./output/chu_test_no_hypo_operationnal_prc.pdf",
    "./output/chu_test_no_inter_operationnal_prc.pdf",
    "./output/chu_test_no_hypo_no_inter_operationnal_prc.pdf",
]


labels = ["(a) Entire patient record", "(b) Normotensive set", "(c) Intervention free set", "(d) Normotensive-intervention free set"]
docs = [fitz.open(p) for p in pdfs]
pages = [d[0] for d in docs]

max_w = max(p.rect.width for p in pages)
max_h = max(p.rect.height for p in pages)

label_space = 40
gap = 12

page_width = 2 * max_w + gap
page_height = 2 * (max_h + label_space) + gap + 50

out = fitz.open()
page = out.new_page(width=page_width, height=page_height)

positions = [
    (0, 0),              # top-left
    (max_w + gap, 0),    # top-right
    (0, max_h + label_space + gap),                                      # bottom-left
    (max_w + gap, max_h + label_space + gap),                            # bottom-right
]

for src_page, label, (x, y) in zip(pages, labels, positions):
    # PDF area (above label)
    pdf_rect = fitz.Rect(
        x,
        y + label_space,
        x + max_w,
        y + label_space + max_h,
    )
    page.show_pdf_page(pdf_rect, src_page.parent, src_page.number)

    # Label area (below PDF)
    y+= max_h + 4*gap

    print(f'x={x}, y={y}')
    label_rect = fitz.Rect(
        x,
        y,
        x + max_w,
        y + label_space,
    )
    page.insert_textbox(
        label_rect,
        label,
        fontsize=18,
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER,
    )

out.save("Figure_5.pdf")
out.close()

pdfs = [
    "./output/test_no_hypo_no_inter_operationnal_prc.pdf",
    "./output/test_no_hypo_no_inter_roc.pdf",
    "./output/test_no_hypo_no_inter_prc.pdf",
    "./output/test_no_hypo_no_inter_calibration.pdf",
]


labels = ["(a) Operational PRC", "(b) ROC", "(c) standard PRC", "(d) Calibration curve"]
docs = [fitz.open(p) for p in pdfs]
pages = [d[0] for d in docs]

max_w = max(p.rect.width for p in pages)
max_h = max(p.rect.height for p in pages)

label_space = 40
gap = 12

page_width = 2 * max_w + gap
page_height = 2 * (max_h + label_space) + gap + 50

out = fitz.open()
page = out.new_page(width=page_width, height=page_height)

positions = [
    (0, 0),              # top-left
    (max_w + gap, 0),    # top-right
    (0, max_h + label_space + gap),                                      # bottom-left
    (max_w + gap, max_h + label_space + gap),                            # bottom-right
]

for src_page, label, (x, y) in zip(pages, labels, positions):
    # PDF area (above label)
    pdf_rect = fitz.Rect(
        x,
        y + label_space,
        x + max_w,
        y + label_space + max_h,
    )
    page.show_pdf_page(pdf_rect, src_page.parent, src_page.number)

    # Label area (below PDF)
    y+= max_h + 4*gap

    print(f'x={x}, y={y}')
    label_rect = fitz.Rect(
        x,
        y,
        x + max_w,
        y + label_space,
    )
    page.insert_textbox(
        label_rect,
        label,
        fontsize=18,
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER,
    )

out.save("Figure_6.pdf")
out.close()


pdfs = [
    "./output/chu_test_no_hypo_no_inter_operationnal_prc.pdf",
    "./output/chu_test_no_hypo_no_inter_roc.pdf",
    "./output/chu_test_no_hypo_no_inter_prc.pdf",
    "./output/chu_test_no_hypo_no_inter_calibration.pdf",
]


labels = ["(a) Operational PRC", "(b) ROC", "(c) standard PRC", "(d) Calibration curve"]
docs = [fitz.open(p) for p in pdfs]
pages = [d[0] for d in docs]

max_w = max(p.rect.width for p in pages)
max_h = max(p.rect.height for p in pages)

label_space = 40
gap = 12

page_width = 2 * max_w + gap
page_height = 2 * (max_h + label_space) + gap + 50

out = fitz.open()
page = out.new_page(width=page_width, height=page_height)

positions = [
    (0, 0),              # top-left
    (max_w + gap, 0),    # top-right
    (0, max_h + label_space + gap),                                      # bottom-left
    (max_w + gap, max_h + label_space + gap),                            # bottom-right
]

for src_page, label, (x, y) in zip(pages, labels, positions):
    # PDF area (above label)
    pdf_rect = fitz.Rect(
        x,
        y + label_space,
        x + max_w,
        y + label_space + max_h,
    )
    page.show_pdf_page(pdf_rect, src_page.parent, src_page.number)

    # Label area (below PDF)
    y+= max_h + 4*gap
    print(f'x={x}, y={y}')
    label_rect = fitz.Rect(
        x,
        y,
        x + max_w,
        y + label_space,
    )
    page.insert_textbox(
        label_rect,
        label,
        fontsize=18,
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_CENTER,
    )

out.save("Figure_7.pdf")
out.close()


i=1
for metric in ["roc", "calibration", "prc"]:
    for chu in ["", "chu_"]:
        pdfs = [
            f"./output/{chu}test_full_{metric}.pdf",
            f"./output/{chu}test_no_hypo_{metric}.pdf",
            f"./output/{chu}test_no_inter_{metric}.pdf",
            f"./output/{chu}test_no_hypo_no_inter_{metric}.pdf",
        ]


        labels = ["(a) Entire patient record", "(b) Normotensive set", "(c) Intervention free set", "(d) Normotensive-intervention free set"]
        docs = [fitz.open(p) for p in pdfs]
        pages = [d[0] for d in docs]

        max_w = max(p.rect.width for p in pages)
        max_h = max(p.rect.height for p in pages)

        label_space = 40
        gap = 12

        page_width = 2 * max_w + gap
        page_height = 2 * (max_h + label_space) + gap + 50

        out = fitz.open()
        page = out.new_page(width=page_width, height=page_height)

        positions = [
            (0, 0),              # top-left
            (max_w + gap, 0),    # top-right
            (0, max_h + label_space + gap),                                      # bottom-left
            (max_w + gap, max_h + label_space + gap),                            # bottom-right
        ]

        for src_page, label, (x, y) in zip(pages, labels, positions):
            # PDF area (above label)
            pdf_rect = fitz.Rect(
                x,
                y + label_space,
                x + max_w,
                y + label_space + max_h,
            )
            page.show_pdf_page(pdf_rect, src_page.parent, src_page.number)

            # Label area (below PDF)
            y+= max_h + 4*gap
            print(f'x={x}, y={y}')
            label_rect = fitz.Rect(
                x,
                y,
                x + max_w,
                y + label_space,
            )
            page.insert_textbox(
                label_rect,
                label,
                fontsize=18,
                color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_CENTER,
            )

        out.save(f"Figure_A{i}.pdf")
        i+=1
        out.close()