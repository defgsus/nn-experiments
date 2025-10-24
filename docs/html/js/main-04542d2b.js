"use strict";

document.addEventListener("DOMContentLoaded", function () {
  /* based on https://stackoverflow.com/questions/14267781/sorting-html-table-with-javascript/49041392#49041392 */
  function hookTables() {
    var getCellValue = function getCellValue(tr, idx) {
      var v = tr.children[idx].innerText || tr.children[idx].textContent;

      // remove commas from integers
      v = v.replaceAll(",", "");

      // remove the typical endings
      if (v.endsWith("/s")) v = v.slice(0, v.length - 2);else if (v.endsWith(" sec")) v = v.slice(0, v.length - 4);else if (v.indexOf(" (") > 0) v = v.split("(")[0];
      return v;
    };
    var comparer = function comparer(idx, asc) {
      return function (tr_a, tr_b) {
        var a = getCellValue(asc ? tr_a : tr_b, idx);
        var b = getCellValue(asc ? tr_b : tr_a, idx);
        if (a !== "" && a !== "" && !isNaN(a) && !isNaN(b)) {
          return a - b;
        }
        return a.toString().localeCompare(b);
      };
    };
    document.querySelectorAll("table").forEach(function ($table) {
      $table._original_rows = Array.from($table.querySelector('tbody').querySelectorAll('tr'));
      $table.querySelector("thead").querySelectorAll("th").forEach(function ($th) {
        $th.classList.add("sortable");
        $th.setAttribute("title", "click to sort");
        $th.addEventListener("click", function () {
          var $tbody = $table.querySelector("tbody");
          var index = Array.from($th.parentNode.children).indexOf($th);
          if ($th.classList.contains("sorted") && !$th.classList.contains("ascending")) {
            // return to original order
            $table.querySelector("thead").querySelectorAll("th").forEach(function ($th) {
              $th.classList.remove("sorted");
            });
            $table._original_rows.forEach(function ($tr) {
              return $tbody.appendChild($tr);
            });
          } else {
            // clear other sort headers
            $table.querySelector("thead").querySelectorAll("th").forEach(function ($th) {
              $th.classList.remove("sorted");
            });
            $th.classList.add("sorted");
            $th.classList.toggle("ascending");
            Array.from($table.querySelector('tbody').querySelectorAll('tr')).sort(comparer(index, $th.classList.contains("ascending"))).forEach(function ($tr) {
              return $tbody.appendChild($tr);
            });
          }
        });
      });
    });
  }
  window.nnblog_hook_tables = hookTables;
  hookTables();
  function hookIndexTagSelect() {
    var $div = document.querySelector(".tag-select");
    if (!$div) return;
    var tags_html = $div.getAttribute("data-tags").split("/").map(function (t) {
      return t.split(",");
    }).map(function (t) {
      return "<div class=\"tag\" title=\"".concat(t[1], "\" data-tag=\"").concat(t[0], "\">").concat(t[0], "</div>");
    }).join("");
    $div.innerHTML = "<hr/><div class=\"tags\">".concat(tags_html, "</div><hr/>");
    function set_article_filter(tag) {
      document.querySelectorAll(".article-list .article-item").forEach(function ($elem) {
        if (!tag) {
          $elem.classList.remove("hidden");
        } else {
          var elem_tags = $elem.getAttribute("data-tags").split(".");
          if (elem_tags.indexOf(tag) >= 0) {
            $elem.classList.remove("hidden");
          } else {
            $elem.classList.add("hidden");
          }
        }
      });
    }
    var select_tag = {};
    document.querySelectorAll(".tag-select .tag").forEach(function ($elem) {
      $elem.addEventListener("click", function () {
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
