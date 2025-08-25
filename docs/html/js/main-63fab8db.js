document.addEventListener("DOMContentLoaded", function() {

    /* based on https://stackoverflow.com/questions/14267781/sorting-html-table-with-javascript/49041392#49041392 */
    function hookTables() {
        const getCellValue = (tr, idx) => {
            let v = tr.children[idx].innerText || tr.children[idx].textContent;

            // remove commas from integers
            v = v.replaceAll(",", "");
            // remove the typical endings
            if (v.endsWith("/s"))
                v = v.slice(0, v.length - 2);
            else if (v.endsWith(" sec"))
                v = v.slice(0, v.length - 4);
            return v;
        }

        const comparer = (idx, asc) => (tr_a, tr_b) => {
            let a = getCellValue(asc ? tr_a : tr_b, idx);
            let b = getCellValue(asc ? tr_b : tr_a, idx);
            if (a !== "" && a !== "" && !isNaN(a) && !isNaN(b)) {
                return a - b;
            }
            return a.toString().localeCompare(b);
        };

        document.querySelectorAll("table").forEach($table => {
            $table._original_rows = Array.from($table.querySelector('tbody').querySelectorAll('tr'));

            $table.querySelector("thead").querySelectorAll("th").forEach($th => {
                $th.classList.add("sortable");
                $th.setAttribute("title", "click to sort");

                $th.addEventListener("click", () => {
                    const $tbody = $table.querySelector("tbody");
                    const index = Array.from($th.parentNode.children).indexOf($th);

                    if ($th.classList.contains("sorted") && !$th.classList.contains("ascending")) {
                        // return to original order
                        $table.querySelector("thead").querySelectorAll("th").forEach($th => {
                            $th.classList.remove("sorted");
                        });
                        $table._original_rows.forEach($tr => $tbody.appendChild($tr));
                    }
                    else {
                        // clear other sort headers
                        $table.querySelector("thead").querySelectorAll("th").forEach($th => {
                            $th.classList.remove("sorted");
                        });

                        $th.classList.add("sorted");
                        $th.classList.toggle("ascending");

                        Array.from($table.querySelector('tbody').querySelectorAll('tr'))
                            .sort(comparer(index, $th.classList.contains("ascending")))
                            .forEach($tr => $tbody.appendChild($tr));
                    }
                });
            })
        });
    }

    hookTables();

    function hookIndexTagSelect() {
        const $div = document.querySelector(".tag-select");
        if (!$div) return;

        const tags_html =
            $div.getAttribute("data-tags").split("/")
                .map(t => t.split(","))
                .map(t => `<div class="tag" title="${t[1]}" data-tag="${t[0]}">${t[0]}</div>`)
                .join("");
        $div.innerHTML = `<hr/><div class="tags">${tags_html}</div><hr/>`;

        function set_article_filter(tag) {
            document.querySelectorAll(".article-list .article-item").forEach($elem => {
                if (!tag) {
                    $elem.classList.remove("hidden");
                } else {
                    const elem_tags = $elem.getAttribute("data-tags").split(".");
                    if (elem_tags.indexOf(tag) >= 0) {
                        $elem.classList.remove("hidden");
                    } else {
                        $elem.classList.add("hidden");
                    }
                }
            });
        }

        const select_tag = {};

        document.querySelectorAll(".tag-select .tag").forEach($elem => {
            $elem.addEventListener("click", () => {
                if (select_tag.elem) {
                    select_tag.elem.classList.remove("selected");
                    set_article_filter(null);
                }
                if ($elem === select_tag.elem) {
                    select_tag.elem = null;
                    set_article_filter(null);
                } else {
                    select_tag.elem = $elem;
                    select_tag.elem.classList.add("selected");
                    set_article_filter($elem.getAttribute("data-tag"));
                }
            });
        });
    }

    hookIndexTagSelect();
});