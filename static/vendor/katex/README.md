# KaTeX 0.18.4 (vendored)

Math rendering for `static/index.html`. Vendored rather than pulled from a CDN so
the page keeps working with no external network dependency.

From the npm tarball `https://registry.npmjs.org/katex/-/katex-0.18.4.tgz`,
verified against the registry's published integrity hash:

    sha512-IMPntbRLOU+eu88XDiFKqQ8Akhr9Tv7jDMXqPhjG9SI1JMA4DIgXk4x9k4skJz2NZJXBRbC+2pYBLj9olqcZow==

Copied unmodified from `package/dist/`:

    katex.min.css
    katex.min.js
    contrib/auto-render.min.js
    fonts/*.woff2

Only the `.woff2` fonts are kept (20 of the 60 font files, ~600K total). Each
`@font-face` in `katex.min.css` lists woff2 first, so a browser that supports
woff2 never requests the `.woff`/`.ttf` fallbacks — those URLs stay in the CSS
but are never fetched. Every browser that supports the `<dialog>` element this
page relies on also supports woff2.

To upgrade: download the new tarball, check its hash against the registry, and
replace the files above. The CSS expects `fonts/` alongside it.
