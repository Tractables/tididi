// The ordered entries come from the walkthrough declarations in guide.rs.
function showExampleNavigation() {
    const sidebar = document.getElementById("rustdoc-modnav");
    if (!sidebar) return;
    const isIndex = /\/guide\/examples\/(?:index\.html)?$/.test(location.pathname);
    const base = new URL(isIndex ? "./" : "../", location.href);
    const heading = document.createElement("h2");
    const index = document.createElement("a");
    index.href = new URL("index.html", base).href;
    index.textContent = "Examples";
    heading.append(index);
    const list = document.createElement("ol");
    list.className = "block example-reading-order";
    list.style.paddingLeft = "1.6em";
    list.style.listStyle = "decimal";
    for (const [name, title] of examples) {
        const item = document.createElement("li");
        item.style.display = "list-item";
        item.style.listStyle = "decimal";
        const link = document.createElement("a");
        link.href = new URL(`${name}/index.html`, base).href;
        link.textContent = title;
        if (new URL(link.href).pathname === location.pathname.replace(/\/$/, "/index.html")) {
            link.setAttribute("aria-current", "page");
            link.style.fontWeight = "600";
        }
        item.append(link);
        list.append(item);
    }
    sidebar.replaceChildren(heading, list);
    if (isIndex) {
        // The numbered index already links every example.
        const modules = document.getElementById("modules");
        if (modules?.nextElementSibling?.classList.contains("item-table")) {
            modules.nextElementSibling.remove();
            modules.remove();
        }
        for (const link of document.querySelectorAll('#rustdoc-toc a[href="#modules"]')) {
            const section = link.closest("ul, h3");
            if (section) section.remove();
        }
    }
}
if (document.readyState === "complete") showExampleNavigation();
else window.addEventListener("load", showExampleNavigation, { once: true });
