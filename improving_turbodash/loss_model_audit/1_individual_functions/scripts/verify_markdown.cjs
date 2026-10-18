// Reproduce equation and link checks using the installed VS Code Markdown+Math.
// Usage: node scripts/verify_markdown.cjs [mdmath/node_modules]
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const base = path.resolve(__dirname, ".."), audit = path.dirname(base);
const repo = path.resolve(base, "../../..");
let modules = process.argv[2];
if (!modules) {
  const extensions = path.join(process.env.USERPROFILE || process.env.HOME, ".vscode", "extensions");
  const versions = fs.readdirSync(extensions).filter(n => n.startsWith("goessner.mdmath-")).sort();
  if (!versions.length) throw new Error("Pass the Markdown+Math node_modules directory.");
  modules = path.join(extensions, versions.at(-1), "node_modules");
}
const katex = require(path.join(modules, "katex"));
const MarkdownIt = require(path.join(modules, "markdown-it"));
const texmath = require(path.join(modules, "markdown-it-texmath"));
const md = new MarkdownIt({html:true}).use(texmath, {engine:katex, delimiters:"dollars"});
function markdownFiles(directory) {
  return fs.readdirSync(directory, {withFileTypes:true}).flatMap(e => {
    if (e.name.startsWith(".") || e.name === "data") return [];
    const filename = path.join(directory, e.name);
    return e.isDirectory() ? markdownFiles(filename) : e.name.endsWith(".md") ? [filename] : [];
  });
}
const filenames = [...markdownFiles(audit),
  path.join(repo, "improving_turbodash/ainley_mathieson_fitting/readme.md"),
  path.join(repo, "improving_turbodash/ainley_mathieson_interpolation/readme.md")];
const files = [], errors = [];
for (const filename of filenames) {
  const source = fs.readFileSync(filename, "utf8");
  const file = path.relative(repo, filename).replaceAll("\\", "/");
  const counts = {inline_equations:0, display_equations:0, local_links:0};
  function walk(tokens) {
    for (const t of tokens) {
      if (t.type.startsWith("math_") && !t.type.endsWith("_end")) {
        const display = t.type !== "math_inline";
        counts[display ? "display_equations" : "inline_equations"]++;
        try { katex.renderToString(t.content, {throwOnError:true, strict:"error", displayMode:display}); }
        catch(e) { errors.push({file, math:t.content, error:e.message}); }
      }
      if (t.type === "text" && (t.content.includes("$") || /\\[()[\]]/.test(t.content)))
        errors.push({file, unparsed_math:t.content});
      if (t.type === "image" || t.type === "link_open") {
        const href = t.attrGet(t.type === "image" ? "src" : "href");
        if (href && !/^[a-z]+:/i.test(href)) {
          counts.local_links++;
          const name = decodeURIComponent(href.split("#")[0]);
          if (name && !fs.existsSync(path.resolve(path.dirname(filename), name)))
            errors.push({file, missing:href});
        }
      }
      if (t.children) walk(t.children);
    }
  }
  walk(md.parse(source, {}));
  if (md.render(source).includes("katex-error")) errors.push({file, html:"KaTeX rendering error"});
  files.push({file, ...counts, sha256:crypto.createHash("sha256").update(source).digest("hex")});
}
const result = {passed:errors.length===0, renderer:"Installed VS Code Markdown+Math, dollar delimiters",
  katex_version:katex.version, modules, markdown_files:files.length,
  inline_equations:files.reduce((s,f)=>s+f.inline_equations,0),
  display_equations:files.reduce((s,f)=>s+f.display_equations,0),
  local_links:files.reduce((s,f)=>s+f.local_links,0), files, errors};
fs.writeFileSync(path.join(base, "data/validation/markdown_math.json"), JSON.stringify(result,null,2)+"\n");
console.log(JSON.stringify({...result, files:undefined},null,2));
if (errors.length) process.exitCode = 1;
