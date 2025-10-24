"use strict";

function _createForOfIteratorHelper(r, e) { var t = "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"]; if (!t) { if (Array.isArray(r) || (t = _unsupportedIterableToArray(r)) || e && r && "number" == typeof r.length) { t && (r = t); var _n = 0, F = function F() {}; return { s: F, n: function n() { return _n >= r.length ? { done: !0 } : { done: !1, value: r[_n++] }; }, e: function e(r) { throw r; }, f: F }; } throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); } var o, a = !0, u = !1; return { s: function s() { t = t.call(r); }, n: function n() { var r = t.next(); return a = r.done, r; }, e: function e(r) { u = !0, o = r; }, f: function f() { try { a || null == t["return"] || t["return"](); } finally { if (u) throw o; } } }; }
function _slicedToArray(r, e) { return _arrayWithHoles(r) || _iterableToArrayLimit(r, e) || _unsupportedIterableToArray(r, e) || _nonIterableRest(); }
function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _unsupportedIterableToArray(r, a) { if (r) { if ("string" == typeof r) return _arrayLikeToArray(r, a); var t = {}.toString.call(r).slice(8, -1); return "Object" === t && r.constructor && (t = r.constructor.name), "Map" === t || "Set" === t ? Array.from(r) : "Arguments" === t || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t) ? _arrayLikeToArray(r, a) : void 0; } }
function _arrayLikeToArray(r, a) { (null == a || a > r.length) && (a = r.length); for (var e = 0, n = Array(a); e < a; e++) n[e] = r[e]; return n; }
function _iterableToArrayLimit(r, l) { var t = null == r ? null : "undefined" != typeof Symbol && r[Symbol.iterator] || r["@@iterator"]; if (null != t) { var e, n, i, u, a = [], f = !0, o = !1; try { if (i = (t = t.call(r)).next, 0 === l) { if (Object(t) !== t) return; f = !1; } else for (; !(f = (e = i.call(t)).done) && (a.push(e.value), a.length !== l); f = !0); } catch (r) { o = !0, n = r; } finally { try { if (!f && null != t["return"] && (u = t["return"](), Object(u) !== u)) return; } finally { if (o) throw n; } } return a; } }
function _arrayWithHoles(r) { if (Array.isArray(r)) return r; }
document.addEventListener("DOMContentLoaded", function () {
  var $root_elem = document.querySelector("#interest-graph-ui");
  var $info = $root_elem.querySelector(".interest-graph-ui-info");
  var $input = $root_elem.querySelector(".interest-graph-ui-input");
  var $result = $root_elem.querySelector(".interest-graph-ui-result");
  var $result_num = $root_elem.querySelector(".interest-graph-ui-num");
  var $feature_select = $root_elem.querySelector(".interest-graph-ui-select");
  var $map = $root_elem.querySelector(".interest-graph-ui-map");
  var graph_data = null;
  function on_data_loaded(data) {
    $result.innerText = "Data Loaded!";
    graph_data = data;
    window.graph_data = graph_data;
    graph_data.vertex_map = {};
    for (var i in graph_data.vertices) {
      graph_data.vertex_map[graph_data.vertices[i]] = {
        index: i,
        count: graph_data.vertex_counts[i],
        edges: []
      };
    }
    graph_data.edges = graph_data.edges.map(function (_ref) {
      var _ref2 = _slicedToArray(_ref, 3),
        a = _ref2[0],
        b = _ref2[1],
        count = _ref2[2];
      return [graph_data.vertices[a], graph_data.vertices[b], count];
    });
    var _iterator = _createForOfIteratorHelper(graph_data.edges),
      _step;
    try {
      for (_iterator.s(); !(_step = _iterator.n()).done;) {
        var _step$value = _slicedToArray(_step.value, 3),
          a = _step$value[0],
          b = _step$value[1],
          count = _step$value[2];
        graph_data.vertex_map[a].edges.push([b, count]);
        graph_data.vertex_map[b].edges.push([a, count]);
      }
    } catch (err) {
      _iterator.e(err);
    } finally {
      _iterator.f();
    }
    $info.innerText = "profiles: ".concat(graph_data.num_users) + "\ninterests: ".concat(graph_data.num_vertices) + "\ninterest connections: ".concat(graph_data.num_edges);
    $result_num.innerHTML = [10, 50, 100, 500, "all"].map(function (i) {
      return "<option value=\"".concat(i, "\">").concat(i === 'all' ? "all (".concat(graph_data.num_vertices, ")") : i, "</option>");
    });
    $result_num.onchange = function () {
      query_word($input.value);
      set_url_hash($input.value, $result_num.value, $feature_select.value);
    };
    $feature_select.innerHTML = [{
      name: "edge count"
    }].concat(graph_data.vertex_positions).map(function (i) {
      return "<option value=\"".concat(i.name.replaceAll(' ', '_'), "\">").concat(i.name, "</option>");
    });
    $feature_select.value = "magic-mix-tsne2";
    $feature_select.onchange = function () {
      query_word($input.value);
      set_url_hash($input.value, $result_num.value, $feature_select.value);
    };
    if (!update_ui_from_url_hash() && $input.value) {
      query_word($input.value);
    }
  }
  function get_vertices_edges(vertices) {
    var vert_set = new Set(vertices);
    var x = graph_data.edges.filter(function (_ref3) {
      var _ref4 = _slicedToArray(_ref3, 3),
        a = _ref4[0],
        b = _ref4[1],
        count = _ref4[2];
      return vert_set.has(a) & vert_set.has(b);
    });
    return x;
  }
  function calc_distance(pos1, pos2) {
    var dx = pos1[0] - pos2[0];
    var dy = pos1[1] - pos2[1];
    return Math.sqrt(dx * dx + dy * dy);
  }
  function get_closest_vertices(word, feature_name) {
    var v = graph_data.vertex_map[word];
    if (!v) {
      return [];
    }
    var positions = graph_data.vertex_positions.filter(function (i) {
      return i.name === feature_name;
    })[0].data;
    var word_pos = positions[graph_data.vertex_map[word].index];
    var distances = positions.map(function (pos, i) {
      return [calc_distance(pos, word_pos), i];
    });
    distances.sort(function (a, b) {
      return a[0] <= b[0] ? -1 : 1;
    });
    return distances.map(function (d) {
      return [graph_data.vertices[d[1]], Math.round(d[0] * 100) / 100];
    });
  }
  function count_links(word1, word2) {
    var v = graph_data.vertex_map[word1];
    if (!v) return 0;
    var _iterator2 = _createForOfIteratorHelper(v.edges),
      _step2;
    try {
      for (_iterator2.s(); !(_step2 = _iterator2.n()).done;) {
        var _step2$value = _slicedToArray(_step2.value, 2),
          w2 = _step2$value[0],
          count = _step2$value[1];
        if (w2 === word2) return count;
      }
    } catch (err) {
      _iterator2.e(err);
    } finally {
      _iterator2.f();
    }
    return 0;
  }
  function get_closest_vertices_by_edge_count(word) {
    var v = graph_data.vertex_map[word];
    if (!v) {
      return [];
    }
    return [[word, 0]].concat(v.edges);
  }
  function percent(x, n) {
    var p = Math.round(x / n * 10000) / 100;
    return "".concat(p, "%");
  }
  function render_result_table(word, vertices, count, with_distance) {
    if (!vertices || !vertices.length) {
      $result.innerText = "Sorry, '".concat(word, "' is not in the dataset.");
      return;
    }
    var sum_links = graph_data.vertex_map[word].edges.reduce(function (a, e) {
      return a + e[1];
    }, 0);
    var html = "<table class=\"interest-graph-ui-table\"><thead><tr>" + "<th>interest</th>" + "<th>times used</th>" + "<th style=\"max-width: 8rem;\">connections with '".concat(word, "'<br/>(").concat(sum_links, " at all)</th>") + (with_distance ? "<th style=\"max-width: 8rem;\">distance to '".concat(word, "'</th>") : "") + "</tr></thead><tbody>";
    var _iterator3 = _createForOfIteratorHelper(vertices.slice(0, count)),
      _step3;
    try {
      for (_iterator3.s(); !(_step3 = _iterator3.n()).done;) {
        var vert = _step3.value;
        var _count = graph_data.vertex_map[vert[0]].count;
        var num_links = count_links(word, vert[0]);
        html += "<tr class=\"interest-graph-word\" data-word=\"".concat(vert[0], "\">") + "<td >".concat(vert[0], "</td>") + "<td class=\"align-right mono\">".concat(_count, " (").concat(percent(_count, graph_data.num_users), ")</td>") + "<td class=\"align-right mono\">" + (num_links ? "".concat(num_links, " (").concat(percent(num_links, sum_links), ")") : '') + "</td>" + (with_distance ? "<td class=\"align-right mono\">".concat(vert[1] ? vert[1] : '', "</td>") : "") + "</tr>";
      }
    } catch (err) {
      _iterator3.e(err);
    } finally {
      _iterator3.f();
    }
    html += '</tbody></table>';
    $result.innerHTML = html;
    hook_words();
    if (window.nnblog_hook_tables) {
      window.nnblog_hook_tables();
    }
  }
  function mix(a, b, t) {
    return a * (1. - t) + t * b;
  }
  function change_map_zoom(zoom, rx, ry) {
    var $svg = $map.querySelector("svg");
    var _$svg$getAttribute$sp = $svg.getAttribute("viewBox").split(/\s+/).map(parseFloat),
      _$svg$getAttribute$sp2 = _slicedToArray(_$svg$getAttribute$sp, 4),
      vx = _$svg$getAttribute$sp2[0],
      vy = _$svg$getAttribute$sp2[1],
      vw = _$svg$getAttribute$sp2[2],
      vh = _$svg$getAttribute$sp2[3];
    var new_vw = vw * zoom;
    var new_vh = vh * zoom;
    var new_vx = mix(vx, vx - (new_vw - vw), rx);
    var new_vy = mix(vy, vy - (new_vh - vh), ry);
    $svg.setAttribute("viewBox", [new_vx, new_vy, new_vw, new_vh].join(" "));
  }
  ;
  function on_map_wheel(event) {
    event.preventDefault();
    event.stopPropagation();
    var $svg = $map.querySelector("svg");
    var zoom = 1. + .1 * (event.deltaY > 0 ? 1. : -1.);
    var x = event.layerX;
    var y = event.layerY;
    var rx = x / $svg.clientWidth;
    var ry = y / $svg.clientHeight;
    change_map_zoom(zoom, rx, ry);
  }
  var MAP_SCALE = 20.;
  function render_map_svg(word, vertices, feature_name) {
    var positions = graph_data.vertex_positions.filter(function (i) {
      return i.name === feature_name;
    })[0].data;
    var _positions$graph_data = _slicedToArray(positions[graph_data.vertex_map[word].index], 2),
      x = _positions$graph_data[0],
      y = _positions$graph_data[1];
    x = x * MAP_SCALE;
    y = y * MAP_SCALE;
    var pad = 5 * MAP_SCALE;
    var html = "";
    html = "<div class=\"flex\"><div>" + "<svg width=\"400\" height=\"400\" viewBox=\"".concat(x - pad, " ").concat(y - pad, " ").concat(pad * 2, " ").concat(pad * 2, "\" xmlns=\"http://www.w3.org/2000/svg\" style=\"background: #181818; user-select: none;\"><style>") + " .text { font: ".concat(.4 * MAP_SCALE, "px sans-serif; fill: #bbb; cursor: pointer; }") + " .text.highlight { font-weight: bold; fill: #ddd; }" + " .text:hover { text-shadow: 0 0 7px black; fill: #ffffff; }" + " .line { stroke: rgba(100, 150, 200, .3); }" + " .line:hover { stroke: rgba(150, 200, 250, .9); z-index: 10; }" + " </style>";
    var _iterator4 = _createForOfIteratorHelper(graph_data.vertex_map[word].edges),
      _step4;
    try {
      for (_iterator4.s(); !(_step4 = _iterator4.n()).done;) {
        var _step4$value = _slicedToArray(_step4.value, 2),
          word2 = _step4$value[0],
          count = _step4$value[1];
        var _positions$graph_data2 = _slicedToArray(positions[graph_data.vertex_map[word2].index], 2),
          x2 = _positions$graph_data2[0],
          y2 = _positions$graph_data2[1];
        x2 = x2 * MAP_SCALE;
        y2 = y2 * MAP_SCALE;
        html += "<line x1=\"".concat(x, "\" y1=\"").concat(y, "\" x2=\"").concat(x2, "\" y2=\"").concat(y2, "\" class=\"line\">") + "<title>".concat(word, " -> ").concat(word2, "\n(").concat(count, "x)</title>") + "</line>";
      }
      /*for (const [x, y] of positions) {
          html += `<circle cx="${x}" cy="${y}" r=".2" fill="#aaa"/>`;
      }*/
    } catch (err) {
      _iterator4.e(err);
    } finally {
      _iterator4.f();
    }
    for (var i in positions) {
      var _positions$i = _slicedToArray(positions[i], 2),
        _x = _positions$i[0],
        _y = _positions$i[1];
      _x = _x * MAP_SCALE;
      _y = _y * MAP_SCALE;
      var text = graph_data.vertices[i];
      //html += `<circle cx="${x}" cy="${y}" r=".2" fill="#aaa"><title>${text}</title></circle>`;
      html += "<text x=\"".concat(_x, "\" y=\"").concat(_y, "\" class=\"text").concat(text === word ? ' highlight' : '', "\" data-word=\"").concat(text, "\">").concat(text) + "<title>".concat(text, "\n(used ").concat(graph_data.vertex_counts[i], " times)</title></text>");
    }
    html += "</svg></div>" + "<div><div style=\"display: flex; flex-direction: column\">" + "<div><button style=\"width: 24px\" id=\"zoom-button-plus\">+</button></div>" + "<div><button style=\"width: 24px\" id=\"zoom-button-minus\">-</button></div>" + "</div></div>";
    $map.innerHTML = html;
    $map.onwheel = on_map_wheel;
    hook_words();
    $map.querySelector("#zoom-button-plus").onclick = function () {
      change_map_zoom(0.8, .5, .5);
    };
    $map.querySelector("#zoom-button-minus").onclick = function () {
      change_map_zoom(1.2, .5, .5);
    };
  }
  function render_map(word, vertices, feature_name) {
    var html = "";
    if (feature_name !== "edge_count") {
      var min_x = null,
        max_x = null,
        min_y = null,
        max_y = null;
      var items = [];
      var _iterator5 = _createForOfIteratorHelper(vertices.slice(0, 50)),
        _step5;
      try {
        for (_iterator5.s(); !(_step5 = _iterator5.n()).done;) {
          var vert = _step5.value;
          var label = vert[0];
          var vertex = graph_data.vertex_map[label];
          var positions = graph_data.vertex_positions.filter(function (i) {
            return i.name === feature_name;
          })[0].data;
          var pos = positions[vertex.index];
          items.push({
            label: label,
            x: pos[0],
            y: pos[1],
            dist: vert[1]
          });
          if (min_x === null || pos[0] < min_x) min_x = pos[0];
          if (min_y === null || pos[1] < min_y) min_y = pos[1];
          if (max_x === null || pos[0] > max_x) max_x = pos[0];
          if (max_y === null || pos[1] > max_y) max_y = pos[1];
        }
      } catch (err) {
        _iterator5.e(err);
      } finally {
        _iterator5.f();
      }
      var _iterator6 = _createForOfIteratorHelper(vertices.slice(30)),
        _step6;
      try {
        for (_iterator6.s(); !(_step6 = _iterator6.n()).done;) {
          var _vert = _step6.value;
          var _label = _vert[0];
          var _vertex = graph_data.vertex_map[_label];
          var _positions = graph_data.vertex_positions.filter(function (i) {
            return i.name === feature_name;
          })[0].data;
          var _pos = _positions[_vertex.index];
          if (_pos[0] >= min_x && _pos[0] <= max_x && _pos[1] >= min_y && _pos[1] <= max_y) {
            items.push({
              label: _label,
              x: _pos[0],
              y: _pos[1],
              dist: _vert[1]
            });
          }
        }
      } catch (err) {
        _iterator6.e(err);
      } finally {
        _iterator6.f();
      }
      var rect = $map.getBoundingClientRect();
      var pad = .1;
      for (var _i = 0, _items = items; _i < _items.length; _i++) {
        var item = _items[_i];
        var x = ((item.x - min_x) / (max_x - min_x) * (1. - pad) + pad / 2) * rect.width;
        var y = rect.height - 1 - ((item.y - min_y) / (max_y - min_y) * (1. - pad) + pad / 2) * rect.height;
        var title = item.label;
        if (item.label !== word) {
          title = "".concat(item.label, "\ndistance to '").concat(word, "': ").concat(item.dist);
        }
        html += "<div class=\"interest-graph-ui-map-item interest-graph-word ".concat(item.label === word ? 'highlight' : '', "\" data-word=\"").concat(item.label, "\"") + " style=\"position: absolute; top: ".concat(y, "px; left: ").concat(x, "px\" title=\"").concat(title, "\">").concat(item.label, "</div>");
      }
    }
    $map.innerHTML = html;
    hook_words();
  }
  function escape_html(str) {
    var p = document.createElement("p");
    p.appendChild(document.createTextNode(str));
    return p.innerHTML;
  }
  function query_word(word) {
    word = escape_html(word);
    $input.value = word;
    var feature_name = $feature_select.value;
    var vertices = feature_name === "edge_count" ? get_closest_vertices_by_edge_count(word) : get_closest_vertices(word, feature_name);
    var count = $result_num.value === "all" ? graph_data.num_vertices : parseInt($result_num.value);
    if (feature_name !== "edge_count") {
      $map.removeAttribute("hidden");
    } else {
      $map.setAttribute("hidden", "hidden");
    }
    render_result_table(word, vertices, count, feature_name !== "edge_count");
    render_map_svg(word, vertices, feature_name);
  }
  function set_url_hash(word, count, feature_name) {
    //window.history.pushState({}, "", window.location);
    if (count === graph_data.num_vertices) count = "all";
    var param = new URLSearchParams({
      w: word,
      n: count,
      f: feature_name
    });
    window.location.hash = "#".concat(param.toString());
  }
  function update_ui_from_url_hash() {
    var p = new URLSearchParams(window.location.hash.slice(1));
    var update = false;
    if (p.get("w")) {
      $input.value = p.get("w");
      update = true;
    }
    if (p.get("n")) {
      $result_num.value = p.get("n");
      update = true;
    }
    if (p.get("f")) {
      $feature_select.value = p.get("f");
      update = true;
    }
    if (update && $input.value) {
      query_word($input.value);
      return true;
    }
  }
  function scroll_to_ui() {
    var rect = $root_elem.getBoundingClientRect();
    if (!(rect.top >= 0 && rect.left >= 0 && rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && rect.right <= (window.innerWidth || document.documentElement.clientWidth))) {
      $root_elem.scrollIntoView();
    }
  }
  function hook_words() {
    document.querySelectorAll('[data-word]').forEach(function ($elem) {
      $elem.onclick = function (e) {
        e.stopPropagation();
        e.preventDefault();
        scroll_to_ui();
        query_word($elem.getAttribute("data-word"));
        set_url_hash($input.value, $result_num.value, $feature_select.value);
      };
    });
  }
  function hook_links() {
    document.querySelectorAll("a").forEach(function ($elem) {
      var _$elem$getAttribute;
      if (((_$elem$getAttribute = $elem.getAttribute("href")) === null || _$elem$getAttribute === void 0 ? void 0 : _$elem$getAttribute.indexOf("=")) > 0) {
        $elem.onclick = function (e) {
          scroll_to_ui();
        };
      }
    });
  }
  ;
  function hook_ui() {
    hook_links();
    $input.onchange = function () {
      query_word($input.value);
      set_url_hash($input.value, $result_num.value, $feature_select.value);
    };
    window.addEventListener("hashchange", function (e) {
      update_ui_from_url_hash();
    });
    $info.innerHTML = "Loading data ...";
    $root_elem.removeAttribute("hidden");
    fetch("/nn-experiments/posts/2025/interest-graph.json").then(function (r) {
      return r.json();
    })["catch"](function (e) {
      $result.innerText = "Sorry, could not load the data.";
    }).then(function (data) {
      if (!data) throw Error("The XHR requests was probably blocked.");
      return data;
    }).then(on_data_loaded)["catch"](function (e) {
      $result.innerText = "Sorry, something went wrong: ".concat(e.message);
    });
  }
  hook_ui();
});
