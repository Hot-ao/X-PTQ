"""d3: like d2 but text, lines and grays stay gray; data/accent colours use the navy/blue palette (091540, 1B2CC1, 7692FF, ABD2FA). Only colour values change."""
import re, sys, zipfile
src, out = sys.argv[1], sys.argv[2]
M = {
  # text, lines and neutrals stay gray (d3); only the 6B6A64 warm gray becomes neutral
  "6B6A64": "6B6B6B",
  # heat scale (low -> high)
  "F3E7CB": "EEF4FE", "F3EBD9": "EEF4FE", "F3E2BC": "E6EFFD", "F4DEAD": "D6E7FC", "F4D590": "C3DCFB",
  "F5CC73": "ABD2FA", "F0BE73": "93B3FD", "E09473": "7692FF", "CC5C73": "1B2CC1",
  # component colours
  "E3A33B": "7692FF", "F6DCE2": "E3E7FB",            # S* (periwinkle), L_G light fill
  "B03A2E": "091540", "FBE9E7": "E3E7FB",            # emphasis / Before -> navy
  "3B7D5B": "1B2CC1", "E3F0E8": "DCE9FD",            # After -> blue
}
zin = zipfile.ZipFile(src)
x = zin.read("ppt/slides/slide1.xml").decode("utf-8")
x = re.sub(r'(srgbClr val=")([0-9A-Fa-f]{6})"', lambda m: m.group(1) + M.get(m.group(2).upper(), m.group(2)) + '"', x)
# dark-filled shapes (high heat cells) get white text
def whiten(m):
    sp = m.group(0)
    sppr = re.search(r"<p:spPr>.*?</p:spPr>", sp, re.S)
    if sppr and re.search(r'^<p:spPr>(?:(?!<a:ln).)*<a:solidFill><a:srgbClr val="(1B2CC1|091540)"', sppr.group(0), re.S) and "<p:txBody>" in sp:
        head, body = sp.split("<p:txBody>", 1)
        body = body.replace('srgbClr val="222222"', 'srgbClr val="FFFFFF"')
        return head + "<p:txBody>" + body
    return sp
x, n = re.subn(r"<p:sp>.*?</p:sp>", whiten, x, flags=re.S)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zin.infolist():
        if it.filename.endswith("/"): continue
        zo.writestr(it, x.encode("utf-8") if it.filename == "ppt/slides/slide1.xml" else zin.read(it.filename))
left = sorted(set(re.findall(r'srgbClr val="([0-9A-F]{6})"', x)) - set(M.values()) - {"FFFFFF"})
print("shapes scanned", n, "| unmapped colours:", left)
