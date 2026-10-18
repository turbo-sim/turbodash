// Check with the user's installed VS Code Markdown+Math parser and KaTeX.
// Usage: node verify_markdown.cjs [path/to/goessner.mdmath-*/node_modules]
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
let modules = process.argv[2];
if (!modules) {
  const extensions = path.join(process.env.USERPROFILE || process.env.HOME, ".vscode", "extensions");
  const names = fs.readdirSync(extensions).filter(n => n.startsWith("goessner.mdmath-")).sort();
  if (!names.length) throw new Error("Pass the node_modules directory containing markdown-it, markdown-it-texmath and katex.");
  modules = path.join(extensions, names.at(-1), "node_modules");
}
const katex = require(path.join(modules, "katex"));
const MarkdownIt = require(path.join(modules, "markdown-it"));
const texmath = require(path.join(modules, "markdown-it-texmath"));
const md = new MarkdownIt({html:true}).use(texmath, {engine:katex, delimiters:"dollars"});
const filename = path.join(__dirname, "readme.md");
const source = fs.readFileSync(filename, "utf8");
const counts = {inline:0, display:0, local_links:0};
const errors = [];
function walk(tokens) {
  for (const t of tokens) {
    if (t.type.startsWith("math_") && !t.type.endsWith("_end")) {
      const display = t.type !== "math_inline";
      counts[display ? "display" : "inline"]++;
      try { katex.renderToString(t.content, {throwOnError:true, strict:"error", displayMode:display}); }
      catch(e) { errors.push({math:t.content, error:e.message}); }
    }
    if (t.type === "text" && (t.content.includes("$") || /\\[()[\]]/.test(t.content))) {
      errors.push({unparsed_math:t.content});
    }
    if (t.type === "image" || t.type === "link_open") {
      const href = t.attrGet(t.type === "image" ? "src" : "href");
      if (href && !/^[a-z]+:\/\//i.test(href)) {
        counts.local_links++;
        const name = decodeURIComponent(href.split("#")[0]);
        if (name && !fs.existsSync(path.resolve(__dirname, name))) errors.push({missing:href});
      }
    }
    if (t.children) walk(t.children);
  }
}
walk(md.parse(source, {}));
const html = md.render(source);
if (html.includes("katex-error")) errors.push({html:"KaTeX rendering error"});
const result = {passed:errors.length===0, parser:"VS Code Markdown+Math / markdown-it-texmath",
  katex_version:katex.version, modules, counts, errors,
  readme_sha256:crypto.createHash("sha256").update(source).digest("hex")};
fs.writeFileSync(path.join(__dirname, "data", "markdown_validation.json"), JSON.stringify(result,null,2)+"\n");
// Self-contained local HTML view of the math layout; figure paths stay relative.
const cache = path.join(__dirname, ".cache");
fs.mkdirSync(cache,{recursive:true});
const css = path.join(modules,"katex","dist","katex.min.css").replaceAll("\\","/");
fs.writeFileSync(path.join(cache,"readme.html"),
  '<!doctype html><meta charset="utf-8"><base href="../"><link rel="stylesheet" href="file:///'+css+
  '"><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto}img{max-width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}.katex-display{overflow:auto}</style>'+html);
console.log(JSON.stringify(result,null,2));
if (errors.length) process.exitCode=1;
