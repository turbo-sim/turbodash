// Validate the study note with the user's installed VS Code Markdown+Math engine.
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const base = __dirname;
const extensions = path.join(process.env.USERPROFILE || process.env.HOME, '.vscode', 'extensions');
const versions = fs.readdirSync(extensions).filter(v => v.startsWith('goessner.mdmath-')).sort();
const modules = path.join(extensions, versions.at(-1), 'node_modules');
const katex = require(path.join(modules, 'katex'));
const MarkdownIt = require(path.join(modules, 'markdown-it'));
const texmath = require(path.join(modules, 'markdown-it-texmath'));
const md = new MarkdownIt({html:true}).use(texmath, {engine:katex, delimiters:'dollars'});
const source = fs.readFileSync(path.join(base, 'readme.md'), 'utf8');
const errors = [];
let inline = 0, display = 0, links = 0;
function walk(tokens) {
  for (const token of tokens) {
    if (token.type.startsWith('math_') && !token.type.endsWith('_end')) {
      const block = token.type !== 'math_inline';
      if (block) display++; else inline++;
      try { katex.renderToString(token.content, {throwOnError:true, strict:'error', displayMode:block}); }
      catch (e) { errors.push({math:token.content, error:e.message}); }
    }
    if (token.type === 'text' && (token.content.includes('$') || /\\[()[\]]/.test(token.content)))
      errors.push({unparsed_math:token.content});
    if (token.type === 'link_open' || token.type === 'image') {
      const href = token.attrGet(token.type === 'image' ? 'src' : 'href');
      if (href && !/^[a-z]+:/i.test(href)) {
        links++;
        const target = decodeURIComponent(href.split('#')[0]);
        if (target && !fs.existsSync(path.resolve(base, target))) errors.push({missing:target});
      }
    }
    if (token.children) walk(token.children);
  }
}
walk(md.parse(source, {}));
if (md.render(source).includes('katex-error')) errors.push({render_error:true});
const result = {passed:errors.length===0, inline_equations:inline, display_equations:display,
                links, errors, katex_version:katex.version,
                readme_sha256:crypto.createHash('sha256').update(source).digest('hex')};
fs.writeFileSync(path.join(base, 'data/markdown_math.json'), JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
if (errors.length) process.exitCode = 1;
