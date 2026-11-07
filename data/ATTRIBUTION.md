# Picture attribution

The 24 pictures in `frontend/images/` are from [Twemoji](https://github.com/jdecked/twemoji) 15.1.0,
copyright Twitter, Inc. and other contributors, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Each item in `data/items.json` records its
codepoint and source URL. The files are unmodified.

MultiPic (Duñabeitia et al., 2018, https://www.bcbl.eu/databases/multipic) was the preferred source
because it provides naming norms, but its download page could not be reached non-interactively from
the build environment and its license terms were not confirmed. To switch, drop MultiPic images into
`frontend/images/`, update `image_path`, `source` and `accepted_alternatives` per item, and keep this
file in sync. Naming agreement from healthy participants should not be read as clinical difficulty.
