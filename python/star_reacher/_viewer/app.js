/* star_reacher FR-19 playback viewer application.
 *
 * Runs inside the generated single-file HTML: reads the srview JSON block,
 * imports the inlined three.js module through a Blob URL (keeps the vendored
 * file byte-verbatim and the page free of external requests), and renders
 * log playback. Every quantity shown comes from the log's embedded view
 * stream; interpolation between keyframes is display-only and non-physical.
 */
'use strict';
(function () {
  var bootErrorEl = document.getElementById('boot-error');
  function bootError(message) {
    bootErrorEl.textContent = 'viewer failed to start: ' + message;
    bootErrorEl.style.display = 'block';
  }
  var DATA;
  try {
    DATA = JSON.parse(document.getElementById('srview-data').textContent);
  } catch (err) {
    bootError('embedded data block is unreadable: ' + err);
    return;
  }
  // Blob-URL import keeps the vendored ES module byte-identical to upstream
  // (its pinned SHA-256 stays verifiable) while still loading from this one
  // file: a blob: URL is same-document memory, not a network request.
  var threeSource = document.getElementById('three-src').textContent;
  var blobUrl = URL.createObjectURL(
    new Blob([threeSource], { type: 'text/javascript' })
  );
  import(blobUrl).then(
    function (THREE) {
      URL.revokeObjectURL(blobUrl);
      try {
        main(THREE, DATA);
      } catch (err) {
        bootError(err && err.stack ? err.stack : String(err));
      }
    },
    function (err) {
      bootError('three.js module import failed: ' + err);
    }
  );

  function main(THREE, D) {
    // ---- view stream ----------------------------------------------------
    var KM = 1e-3; // scene units are km: keeps float32 GPU coordinates well
    var N = D.frames.t_s.length; // inside precision at orbital magnitudes
    var ts = new Float64Array(D.frames.t_s);
    var rk = new Float64Array(3 * N); // keyframe positions [km]
    var vk = new Float64Array(3 * N); // keyframe velocities [m/s]
    var qk = new Float64Array(4 * N); // keyframe quaternions q_i2b [w,x,y,z]
    for (var i = 0; i < N; i++) {
      rk[3 * i] = D.frames.r_m[i][0] * KM;
      rk[3 * i + 1] = D.frames.r_m[i][1] * KM;
      rk[3 * i + 2] = D.frames.r_m[i][2] * KM;
      vk[3 * i] = D.frames.v_mps[i][0];
      vk[3 * i + 1] = D.frames.v_mps[i][1];
      vk[3 * i + 2] = D.frames.v_mps[i][2];
      qk[4 * i] = D.frames.q_i2b[i][0];
      qk[4 * i + 1] = D.frames.q_i2b[i][1];
      qk[4 * i + 2] = D.frames.q_i2b[i][2];
      qk[4 * i + 3] = D.frames.q_i2b[i][3];
    }
    var t0 = D.epoch.t_first_s;
    var tEnd = D.epoch.t_last_s;
    var bodyRkm = D.body.radius_m === null ? null : D.body.radius_m * KM;
    var rot = D.body.rotation;
    function eraAt(t) {
      return rot.model === 'era' ? rot.era0_rad + rot.rate_radps * t : 0.0;
    }

    // ---- playback state ---------------------------------------------------
    var state = {
      t: t0,
      playing: false,
      speed: 1.0,
      camMode: 'orbit',
      seg: 0 // cached keyframe segment index (playback is nearly sequential)
    };

    function segmentFor(t) {
      var s = state.seg;
      if (s < 0 || s > N - 2) s = 0;
      if (ts[s] <= t && t <= ts[s + 1]) return s;
      var lo = 0;
      var hi = N - 2;
      while (lo < hi) {
        var mid = (lo + hi + 1) >> 1;
        if (ts[mid] <= t) lo = mid;
        else hi = mid - 1;
      }
      return lo;
    }

    // Interpolated sample at t: position/velocity lerp, attitude slerp.
    // Display-only: these are NOT the propagated dynamics between samples.
    var sample = {
      r: new THREE.Vector3(),
      v: new THREE.Vector3(),
      q: new THREE.Quaternion()
    };
    function sampleAt(t) {
      if (N === 1) {
        sample.r.set(rk[0], rk[1], rk[2]);
        sample.v.set(vk[0], vk[1], vk[2]);
        sample.q.set(qk[1], qk[2], qk[3], qk[0]);
        return;
      }
      var s = segmentFor(t);
      state.seg = s;
      var u = (t - ts[s]) / (ts[s + 1] - ts[s]);
      if (u < 0) u = 0;
      if (u > 1) u = 1;
      var a = 3 * s;
      var b = 3 * (s + 1);
      sample.r.set(
        rk[a] + u * (rk[b] - rk[a]),
        rk[a + 1] + u * (rk[b + 1] - rk[a + 1]),
        rk[a + 2] + u * (rk[b + 2] - rk[a + 2])
      );
      sample.v.set(
        vk[a] + u * (vk[b] - vk[a]),
        vk[a + 1] + u * (vk[b + 1] - vk[a + 1]),
        vk[a + 2] + u * (vk[b + 2] - vk[a + 2])
      );
      slerpTo(s, u, sample.q);
    }

    function slerpTo(s, u, out) {
      var a = 4 * s;
      var b = a + 4;
      var w0 = qk[a], x0 = qk[a + 1], y0 = qk[a + 2], z0 = qk[a + 3];
      var w1 = qk[b], x1 = qk[b + 1], y1 = qk[b + 2], z1 = qk[b + 3];
      var dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1;
      if (dot < 0) { // shortest arc: q and -q are the same attitude
        w1 = -w1; x1 = -x1; y1 = -y1; z1 = -z1; dot = -dot;
      }
      var c0, c1;
      if (dot > 0.9995) { // nearly parallel: nlerp avoids sin() blowup
        c0 = 1 - u;
        c1 = u;
      } else {
        var th = Math.acos(Math.min(dot, 1));
        var sth = Math.sin(th);
        c0 = Math.sin((1 - u) * th) / sth;
        c1 = Math.sin(u * th) / sth;
      }
      var w = c0 * w0 + c1 * w1;
      var x = c0 * x0 + c1 * x1;
      var y = c0 * y0 + c1 * y1;
      var z = c0 * z0 + c1 * z1;
      var n = Math.sqrt(w * w + x * x + y * y + z * z) || 1;
      // three.js order is (x, y, z, w); the log is Hamilton scalar-first.
      // q_i2b is the project's frame transformation (v_b = q^-1 v q per the
      // rotation kernel); three.js applies quaternions as the active sandwich
      // q v q^-1, which for this convention IS the body->inertial map, so the
      // logged components are used directly, no conjugation.
      out.set(x / n, y / n, z / n, w / n);
    }

    // ---- renderer ---------------------------------------------------------
    var sceneDiv = document.getElementById('scene');
    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch (err) {
      bootError('WebGL is unavailable: ' + err);
      return;
    }
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    sceneDiv.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070c);

    // Extent-derived camera clipping. The near/far ratio is kept near 2e6 so
    // the 24-bit depth buffer still separates the sphere from the coastline
    // and groundtrack lines offset a few tenths of a percent above it.
    var rMaxKm = 0;
    for (i = 0; i < N; i++) {
      var rr = Math.hypot(rk[3 * i], rk[3 * i + 1], rk[3 * i + 2]);
      if (rr > rMaxKm) rMaxKm = rr;
    }
    var sceneScale = Math.max(rMaxKm, bodyRkm || 1, 1);
    var camera = new THREE.PerspectiveCamera(
      50, 1, sceneScale * 3e-5, sceneScale * 60
    );

    function resize() {
      var w = sceneDiv.clientWidth || window.innerWidth;
      var h = sceneDiv.clientHeight || window.innerHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);
    resize();

    // Fixed display lighting (the log carries no sun geometry): a directional
    // key along +x inertial plus ambient fill so the night side stays legible.
    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    var keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
    keyLight.position.set(1, 0.15, 0.35);
    scene.add(keyLight);

    // ---- central body -----------------------------------------------------
    // bodyGroup content is body-fixed; rotation.z = ERA(t) maps it to the
    // inertial scene (r_i = R3(ERA)^T r_bf, an active +z rotation by ERA).
    var bodyGroup = new THREE.Group();
    scene.add(bodyGroup);
    if (bodyRkm !== null) {
      var sphereGeo = new THREE.SphereGeometry(bodyRkm, 96, 64);
      sphereGeo.rotateX(Math.PI / 2); // put the geometry pole on +z (the log frame)
      var sphere = new THREE.Mesh(
        sphereGeo,
        new THREE.MeshLambertMaterial({ map: makeBodyTexture(D.body.name) })
      );
      bodyGroup.add(sphere);
      bodyGroup.add(makeGraticule(bodyRkm * 1.0008));
      if (D.coastline) bodyGroup.add(makeCoastlines(bodyRkm * 1.0015));
    }

    function latLonToVec(latDeg, lonDeg, radius) {
      var lat = latDeg * Math.PI / 180;
      var lon = lonDeg * Math.PI / 180;
      return new THREE.Vector3(
        radius * Math.cos(lat) * Math.cos(lon),
        radius * Math.cos(lat) * Math.sin(lon),
        radius * Math.sin(lat)
      );
    }

    function makeBodyTexture(name) {
      // Procedural latitude-banded shading, canvas-generated at load: an
      // embedded texture with zero payload bytes and no external fetch
      // (D-16). Deliberately neutral; surface detail is not sim data.
      var tones = {
        earth: ['#16324f', '#1d4a6e', '#2a6484', '#1d4a6e', '#16324f'],
        moon: ['#5a5a5e', '#787878', '#8c8c90', '#787878', '#5a5a5e'],
        mars: ['#6e3b26', '#8c4f2e', '#a06438', '#8c4f2e', '#6e3b26']
      };
      var bands = tones[name] || ['#3a4048', '#525a66', '#646e7c', '#525a66', '#3a4048'];
      var canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 256;
      var ctx = canvas.getContext('2d');
      var grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
      for (var k = 0; k < bands.length; k++) {
        grad.addColorStop(k / (bands.length - 1), bands[k]);
      }
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      var tex = new THREE.CanvasTexture(canvas);
      tex.colorSpace = THREE.SRGBColorSpace;
      return tex;
    }

    function makeGraticule(radius) {
      // 30-degree grid as line geometry (not texture) so it aligns exactly
      // with the same lat/lon mapping the coastline and groundtrack use.
      var verts = [];
      var d, lon, lat;
      for (lon = -180; lon < 180; lon += 30) {
        for (d = -88; d < 88; d += 2) {
          pushSeg(verts, latLonToVec(d, lon, radius), latLonToVec(d + 2, lon, radius));
        }
      }
      for (lat = -60; lat <= 60; lat += 30) {
        for (d = 0; d < 360; d += 2) {
          pushSeg(verts, latLonToVec(lat, d, radius), latLonToVec(lat, d + 2, radius));
        }
      }
      var geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
      return new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({ color: 0x8093a8, transparent: true, opacity: 0.18 })
      );
    }

    function pushSeg(verts, a, b) {
      verts.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }

    function makeCoastlines(radius) {
      var verts = [];
      var segs = D.coastline.segments;
      for (var s = 0; s < segs.length; s++) {
        var seg = segs[s];
        for (var p = 0; p + 1 < seg.length; p++) {
          pushSeg(
            verts,
            latLonToVec(seg[p][1], seg[p][0], radius),
            latLonToVec(seg[p + 1][1], seg[p + 1][0], radius)
          );
        }
      }
      var geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
      return new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({ color: 0x9fb8c8, transparent: true, opacity: 0.55 })
      );
    }

    // ---- inertial triad at the body center (part of the axes overlay) ----
    var inertialTriad = makeTriad(sceneScale * 0.4, 0.55);
    scene.add(inertialTriad);

    function makeTriad(length, opacity) {
      var group = new THREE.Group();
      var axes = [
        [1, 0, 0, 0xd06060],
        [0, 1, 0, 0x60c060],
        [0, 0, 1, 0x6080e0]
      ];
      for (var k = 0; k < 3; k++) {
        var dir = new THREE.Vector3(axes[k][0], axes[k][1], axes[k][2]);
        var arrow = new THREE.ArrowHelper(
          dir, new THREE.Vector3(0, 0, 0), length, axes[k][3],
          length * 0.08, length * 0.04
        );
        arrow.line.material.transparent = true;
        arrow.line.material.opacity = opacity;
        arrow.cone.material.transparent = true;
        arrow.cone.material.opacity = opacity;
        group.add(arrow);
      }
      return group;
    }

    // ---- spacecraft marker + body triad -----------------------------------
    var scGroup = new THREE.Group(); // position = interpolated r
    scene.add(scGroup);
    var attGroup = new THREE.Group(); // quaternion = interpolated q (body axes)
    scGroup.add(attGroup);
    var marker = new THREE.Mesh(
      new THREE.OctahedronGeometry(1),
      new THREE.MeshLambertMaterial({ color: 0xe8ecf2, emissive: 0x30343c })
    );
    attGroup.add(marker);
    var bodyTriad = makeTriad(4.0, 0.95);
    attGroup.add(bodyTriad);

    // ---- trail ------------------------------------------------------------
    var trailGeo = new THREE.BufferGeometry();
    var trailPos = new Float32Array(3 * N);
    for (i = 0; i < 3 * N; i++) trailPos[i] = rk[i];
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
    var trail = new THREE.Line(
      trailGeo,
      new THREE.LineBasicMaterial({ color: 0x6ea8dc, transparent: true, opacity: 0.8 })
    );
    trail.frustumCulled = false;
    scene.add(trail);

    // ---- groundtrack ------------------------------------------------------
    // Dense resample of the keyframe stream, expressed in body-fixed
    // coordinates and parented to bodyGroup so it co-rotates with the body.
    var GT_N = 2048;
    var groundtrack = null;
    var subPoint = null;
    if (bodyRkm !== null && N > 1) {
      var gtVerts = new Float32Array(3 * GT_N);
      for (i = 0; i < GT_N; i++) {
        var tg = t0 + ((tEnd - t0) * i) / (GT_N - 1);
        sampleAt(tg);
        var era = eraAt(tg);
        var ce = Math.cos(era);
        var se = Math.sin(era);
        // frame rotation R3(ERA): inertial -> body-fixed components
        var xb = ce * sample.r.x + se * sample.r.y;
        var yb = -se * sample.r.x + ce * sample.r.y;
        var zb = sample.r.z;
        var nn = Math.hypot(xb, yb, zb) || 1;
        var rad = bodyRkm * 1.003;
        gtVerts[3 * i] = (xb / nn) * rad;
        gtVerts[3 * i + 1] = (yb / nn) * rad;
        gtVerts[3 * i + 2] = (zb / nn) * rad;
      }
      var gtGeo = new THREE.BufferGeometry();
      gtGeo.setAttribute('position', new THREE.BufferAttribute(gtVerts, 3));
      groundtrack = new THREE.Line(
        gtGeo,
        new THREE.LineBasicMaterial({ color: 0xd9a75a, transparent: true, opacity: 0.9 })
      );
      groundtrack.frustumCulled = false;
      bodyGroup.add(groundtrack);
      subPoint = new THREE.Mesh(
        new THREE.SphereGeometry(1, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0xd9a75a })
      );
      bodyGroup.add(subPoint);
    }

    // ---- velocity / force arrows -------------------------------------------
    var velArrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1, 0x50c8c8
    );
    scGroup.add(velArrow);

    var FORCE_COLORS = {
      gravity: 0xc0c060, thirdbody: 0x9070d0, srp: 0xe0e0a0, drag: 0x70b8e0,
      aero: 0x70b8e0, thrust: 0xe08040, rcs: 0xe0a0a0, gravgrad: 0xa0d0a0,
      wheel: 0xd0a0d0
    };
    var forceArrows = [];
    var forceMagRange = null;
    if (D.forces) {
      var fmin = Infinity;
      var fmax = 0;
      for (var si = 0; si < D.forces.sources.length; si++) {
        var recs = D.forces.f_b_n[si];
        for (var ri = 0; ri < recs.length; ri++) {
          var mag = Math.hypot(recs[ri][0], recs[ri][1], recs[ri][2]);
          if (mag > 0) {
            if (mag < fmin) fmin = mag;
            if (mag > fmax) fmax = mag;
          }
        }
        var color = FORCE_COLORS[D.forces.sources[si]] || 0xb0b0b0;
        var arrow = new THREE.ArrowHelper(
          new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1, color
        );
        scGroup.add(arrow);
        forceArrows.push(arrow);
      }
      if (fmax > 0) forceMagRange = [fmin, fmax];
    }

    function forceIndexFor(t) {
      // Step-hold at the last force record <= t: the overlay shows logged
      // samples, never interpolated forces.
      var ft = D.forces.t_s;
      var lo = 0;
      var hi = ft.length - 1;
      if (t < ft[0]) return -1;
      while (lo < hi) {
        var mid = (lo + hi + 1) >> 1;
        if (ft[mid] <= t) lo = mid;
        else hi = mid - 1;
      }
      return lo;
    }

    // ---- overlays wiring ----------------------------------------------------
    var overlays = {
      axes: document.getElementById('ov-axes'),
      vel: document.getElementById('ov-vel'),
      forces: document.getElementById('ov-forces'),
      trail: document.getElementById('ov-trail'),
      ground: document.getElementById('ov-ground')
    };
    if (!D.forces) {
      overlays.forces.disabled = true;
      overlays.forces.parentElement.title = 'this log carries no forces group';
    }
    if (bodyRkm === null) {
      overlays.ground.disabled = true;
      overlays.ground.parentElement.title = 'no display radius for this central body';
    }

    var legendEl = document.getElementById('legend');
    function renderLegend(fi) {
      if (!D.forces || !overlays.forces.checked) {
        legendEl.style.display = 'none';
        return;
      }
      var rows = ['<div><b>forces [N]</b> (arrow length log-scaled)</div>'];
      for (var si = 0; si < D.forces.sources.length; si++) {
        var name = D.forces.sources[si];
        var color = FORCE_COLORS[name] || 0xb0b0b0;
        var magText = '-';
        if (fi >= 0) {
          var f = D.forces.f_b_n[si][fi];
          magText = Math.hypot(f[0], f[1], f[2]).toExponential(3);
        }
        rows.push(
          '<div><span class="swatch" style="background:#' +
          color.toString(16).padStart(6, '0') + '"></span>' +
          name + ' ' + magText + '</div>'
        );
      }
      legendEl.innerHTML = rows.join('');
      legendEl.style.display = 'block';
    }

    // ---- HUD ----------------------------------------------------------------
    var hud = {
      utc: document.getElementById('hud-utc'),
      met: document.getElementById('hud-met'),
      alt: document.getElementById('hud-alt'),
      speed: document.getElementById('hud-speed'),
      event: document.getElementById('hud-event')
    };
    var epoch0Ms = Date.parse(D.epoch.utc_first);

    function utcAt(t) {
      // Scrub extremes return the embedded strings VERBATIM: they were
      // derived in the generator from the log header epoch by one exact
      // datetime addition, which is the Phase 5 exit-criterion-2 equality.
      if (t <= D.epoch.t_first_s) return D.epoch.utc_first;
      if (t >= D.epoch.t_last_s) return D.epoch.utc_last;
      var ms = epoch0Ms + (t - D.epoch.t_first_s) * 1000;
      return new Date(ms).toISOString().replace(/\.?0+Z$/, 'Z');
    }

    function fmtMet(t) {
      return t.toFixed(2) + ' s';
    }

    function eventWindowText(t) {
      var et = D.events.t_s;
      var last = -1;
      for (var k = 0; k < et.length; k++) {
        if (et[k] <= t) last = k;
        else break;
      }
      var parts = [];
      if (last >= 0) parts.push(D.events.detail[last]);
      if (last + 1 < et.length) {
        parts.push('next: ' + D.events.detail[last + 1] +
          ' @ ' + et[last + 1].toFixed(1) + ' s');
      }
      return parts.length ? parts.join(' — ') : 'none';
    }

    function updateHud(t) {
      hud.utc.textContent = utcAt(t);
      hud.met.textContent = fmtMet(t);
      var rKm = sample.r.length();
      if (bodyRkm !== null) {
        hud.alt.textContent = (rKm - bodyRkm).toFixed(1) + ' km (geocentric)';
      } else {
        hud.alt.textContent = 'r = ' + rKm.toFixed(1) + ' km';
      }
      var spd = sample.v.length();
      hud.speed.textContent =
        spd >= 1000 ? (spd / 1000).toFixed(3) + ' km/s' : spd.toFixed(1) + ' m/s';
      hud.event.textContent = eventWindowText(t);
    }

    // ---- timeline -----------------------------------------------------------
    var timeline = document.getElementById('timeline');
    var timelineFill = document.getElementById('timeline-fill');
    for (i = 0; i < D.events.t_s.length; i++) {
      var tick = document.createElement('div');
      tick.className = 'tick';
      var frac = tEnd > t0 ? (D.events.t_s[i] - t0) / (tEnd - t0) : 0;
      tick.style.left = (frac * 100).toFixed(3) + '%';
      tick.title = D.events.detail[i] + ' @ t = ' + D.events.t_s[i] + ' s (code ' +
        D.events.code[i] + ')';
      timeline.appendChild(tick);
    }
    var scrubbing = false;
    function scrubTo(clientX) {
      var rect = timeline.getBoundingClientRect();
      var frac = (clientX - rect.left) / rect.width;
      if (frac < 0) frac = 0;
      if (frac > 1) frac = 1;
      state.t = t0 + frac * (tEnd - t0);
    }
    timeline.addEventListener('pointerdown', function (ev) {
      scrubbing = true;
      timeline.setPointerCapture(ev.pointerId);
      scrubTo(ev.clientX);
    });
    timeline.addEventListener('pointermove', function (ev) {
      if (scrubbing) scrubTo(ev.clientX);
    });
    timeline.addEventListener('pointerup', function () {
      scrubbing = false;
    });

    // ---- transport controls ---------------------------------------------------
    var btnPlay = document.getElementById('btn-play');
    function setPlaying(playing) {
      if (state.t >= tEnd && playing) state.t = t0; // replay from the start
      state.playing = playing;
      btnPlay.textContent = playing ? 'Pause' : 'Play';
    }
    btnPlay.addEventListener('click', function () {
      setPlaying(!state.playing);
    });

    var speedSlider = document.getElementById('speed');
    var speedOut = document.getElementById('speed-out');
    function applySpeed() {
      // Logarithmic slider: -1..3 -> 0.1x..1000x (FR-19 speed range).
      state.speed = Math.pow(10, parseFloat(speedSlider.value));
      speedOut.textContent =
        state.speed >= 10 ? state.speed.toFixed(0) + 'x' : state.speed.toFixed(1) + 'x';
    }
    speedSlider.addEventListener('input', applySpeed);
    applySpeed();

    var camSelect = document.getElementById('cam');
    camSelect.addEventListener('change', function () {
      state.camMode = camSelect.value;
    });

    var helpEl = document.getElementById('help');
    document.getElementById('btn-help').addEventListener('click', function () {
      helpEl.hidden = !helpEl.hidden;
    });
    document.getElementById('help-decimation').textContent =
      D.decimation.kept + ' of ' + D.decimation.total +
      ' truth samples kept; measured max position error ' +
      D.decimation.measured_max_error_m.toFixed(3) + ' m against a bound of ' +
      D.decimation.bound_m.toFixed(3) + ' m (= max(100 m, 0.01 % of the ' +
      'position span))';
    document.getElementById('help-provenance').textContent =
      'Run: epoch ' + D.header.epoch_utc + ', central body ' +
      (D.header.central_body || 'unknown') + ', config sha256 ' +
      D.header.config_sha256.slice(0, 12) + '…, master seed ' +
      D.header.master_seed + '.';

    window.addEventListener('keydown', function (ev) {
      if (ev.target && (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT')) {
        if (ev.key !== ' ') return; // keep slider arrow keys native
      }
      switch (ev.key) {
        case ' ':
          ev.preventDefault();
          setPlaying(!state.playing);
          break;
        case '1': camSelect.value = 'orbit'; state.camMode = 'orbit'; break;
        case '2': camSelect.value = 'chase'; state.camMode = 'chase'; break;
        case '3': camSelect.value = 'body'; state.camMode = 'body'; break;
        case '4': camSelect.value = 'nadir'; state.camMode = 'nadir'; break;
        case 'a': overlays.axes.checked = !overlays.axes.checked; break;
        case 'v': overlays.vel.checked = !overlays.vel.checked; break;
        case 'f':
          if (!overlays.forces.disabled) {
            overlays.forces.checked = !overlays.forces.checked;
          }
          break;
        case 't': overlays.trail.checked = !overlays.trail.checked; break;
        case 'g':
          if (!overlays.ground.disabled) {
            overlays.ground.checked = !overlays.ground.checked;
          }
          break;
        case 'h': helpEl.hidden = !helpEl.hidden; break;
        case 'ArrowLeft':
          state.t = Math.max(t0, state.t - 0.005 * (tEnd - t0));
          break;
        case 'ArrowRight':
          state.t = Math.min(tEnd, state.t + 0.005 * (tEnd - t0));
          break;
        default:
          return;
      }
    });

    // ---- camera ----------------------------------------------------------------
    var orbit = { az: 0.7, el: 0.35, dist: sceneScale * 2.6 };
    var followDist = Math.max(sceneScale * 0.06, 1.0);
    var dragging = false;
    var lastX = 0;
    var lastY = 0;
    renderer.domElement.addEventListener('pointerdown', function (ev) {
      dragging = true;
      lastX = ev.clientX;
      lastY = ev.clientY;
      renderer.domElement.setPointerCapture(ev.pointerId);
    });
    renderer.domElement.addEventListener('pointermove', function (ev) {
      if (!dragging || state.camMode !== 'orbit') return;
      orbit.az -= (ev.clientX - lastX) * 0.005;
      orbit.el += (ev.clientY - lastY) * 0.005;
      var cap = Math.PI / 2 - 0.01;
      if (orbit.el > cap) orbit.el = cap;
      if (orbit.el < -cap) orbit.el = -cap;
      lastX = ev.clientX;
      lastY = ev.clientY;
    });
    renderer.domElement.addEventListener('pointerup', function () {
      dragging = false;
    });
    renderer.domElement.addEventListener(
      'wheel',
      function (ev) {
        ev.preventDefault();
        var factor = Math.exp(ev.deltaY * 0.001);
        if (state.camMode === 'orbit') {
          orbit.dist *= factor;
          var minD = (bodyRkm || sceneScale * 0.01) * 1.05;
          if (orbit.dist < minD) orbit.dist = minD;
          if (orbit.dist > sceneScale * 50) orbit.dist = sceneScale * 50;
        } else {
          followDist *= factor;
          var lo = sceneScale * 1e-4; // stays beyond the near plane
          var hi = sceneScale * 5;
          if (followDist < lo) followDist = lo;
          if (followDist > hi) followDist = hi;
        }
      },
      { passive: false }
    );

    var tmpA = new THREE.Vector3();
    var tmpB = new THREE.Vector3();
    var tmpC = new THREE.Vector3();
    function updateCamera() {
      var sc = scGroup.position;
      if (state.camMode === 'orbit') {
        camera.position.set(
          orbit.dist * Math.cos(orbit.el) * Math.cos(orbit.az),
          orbit.dist * Math.cos(orbit.el) * Math.sin(orbit.az),
          orbit.dist * Math.sin(orbit.el)
        );
        camera.up.set(0, 0, 1);
        camera.lookAt(0, 0, 0);
      } else if (state.camMode === 'chase') {
        tmpA.copy(sample.v);
        if (tmpA.lengthSq() === 0) tmpA.set(1, 0, 0);
        tmpA.normalize();
        tmpB.copy(sc).normalize(); // radial out: lift the chase view a little
        camera.position
          .copy(sc)
          .addScaledVector(tmpA, -followDist)
          .addScaledVector(tmpB, followDist * 0.3);
        camera.up.copy(tmpB);
        camera.lookAt(sc);
      } else if (state.camMode === 'body') {
        // Offset and view direction fixed in the spacecraft BODY frame: the
        // camera rotates with the logged attitude (slerp between keyframes).
        tmpA.set(-1, 0, 0.35).normalize().multiplyScalar(followDist)
          .applyQuaternion(sample.q);
        tmpB.set(1, 0, 0).applyQuaternion(sample.q); // body +x, in inertial
        tmpC.set(0, 0, 1).applyQuaternion(sample.q); // body +z as up
        camera.position.copy(sc).add(tmpA);
        camera.up.copy(tmpC);
        tmpB.multiplyScalar(followDist * 4).add(sc);
        camera.lookAt(tmpB);
      } else { // nadir: on the body-center line above the spacecraft
        tmpA.copy(sc);
        if (tmpA.lengthSq() === 0) tmpA.set(1, 0, 0);
        tmpA.normalize();
        camera.position.copy(sc).addScaledVector(tmpA, followDist);
        camera.up.set(0, 0, 1);
        camera.lookAt(0, 0, 0);
      }
    }

    // ---- per-frame update -------------------------------------------------------
    function logScaledLength(mag, base) {
      // Arrow length spans 0.35..1.0 x base across the run's force-magnitude
      // decades: keeps micro-newton and mega-newton sources both visible.
      if (!forceMagRange || mag <= 0) return 0;
      var lo = Math.log(forceMagRange[0]);
      var hi = Math.log(forceMagRange[1]);
      var u = hi > lo ? (Math.log(mag) - lo) / (hi - lo) : 1;
      return (0.35 + 0.65 * u) * base;
    }

    function updateScene() {
      var t = state.t;
      sampleAt(t);
      scGroup.position.copy(sample.r);
      attGroup.quaternion.copy(sample.q);
      bodyGroup.rotation.z = eraAt(t);

      // Constant apparent size for the marker and its vectors: scale with
      // camera distance so the spacecraft stays visible at any zoom.
      var dist = camera.position.distanceTo(scGroup.position);
      var mScale = Math.max(dist * 0.012, sceneScale * 1e-5);
      marker.scale.setScalar(mScale);
      bodyTriad.scale.setScalar(mScale);
      bodyTriad.visible = overlays.axes.checked;
      inertialTriad.visible = overlays.axes.checked;

      trail.visible = overlays.trail.checked;
      if (trail.visible) {
        var s = N === 1 ? 1 : state.seg + 2; // keyframes flown through t
        trailGeo.setDrawRange(0, Math.min(s, N));
      }

      if (groundtrack) {
        groundtrack.visible = overlays.ground.checked;
        subPoint.visible = overlays.ground.checked;
        if (subPoint.visible) {
          var era = eraAt(t);
          var ce = Math.cos(era);
          var se = Math.sin(era);
          var xb = ce * sample.r.x + se * sample.r.y;
          var yb = -se * sample.r.x + ce * sample.r.y;
          var zb = sample.r.z;
          var nn = Math.hypot(xb, yb, zb) || 1;
          subPoint.position.set(
            (xb / nn) * bodyRkm * 1.004,
            (yb / nn) * bodyRkm * 1.004,
            (zb / nn) * bodyRkm * 1.004
          );
          subPoint.scale.setScalar(Math.max(camera.position.length() * 0.004, bodyRkm * 0.004));
        }
      }

      velArrow.visible = overlays.vel.checked && sample.v.lengthSq() > 0;
      if (velArrow.visible) {
        tmpA.copy(sample.v).normalize();
        velArrow.setDirection(tmpA);
        velArrow.setLength(mScale * 8, mScale * 1.6, mScale * 0.8);
      }

      var fi = -1;
      if (D.forces && overlays.forces.checked) fi = forceIndexFor(t);
      for (var k = 0; k < forceArrows.length; k++) {
        var arrow = forceArrows[k];
        if (fi < 0) {
          arrow.visible = false;
          continue;
        }
        var f = D.forces.f_b_n[k][fi];
        var mag = Math.hypot(f[0], f[1], f[2]);
        var len = logScaledLength(mag, mScale * 10);
        if (len <= 0) {
          arrow.visible = false;
          continue;
        }
        // Forces are logged in the BODY frame; rotate to the inertial scene
        // with the same interpolated attitude the triad uses (display-only).
        tmpA.set(f[0], f[1], f[2]).multiplyScalar(1 / mag).applyQuaternion(sample.q);
        arrow.visible = true;
        arrow.setDirection(tmpA);
        arrow.setLength(len, len * 0.18, len * 0.09);
      }
      renderLegend(overlays.forces.checked ? fi : -1);

      updateCamera();
      updateHud(t);
    }

    // ---- animation loop ---------------------------------------------------------
    var lastFrameMs = null;
    function frame(nowMs) {
      if (state.playing && lastFrameMs !== null) {
        state.t += ((nowMs - lastFrameMs) / 1000) * state.speed;
        if (state.t >= tEnd) {
          state.t = tEnd;
          setPlaying(false);
        }
      }
      lastFrameMs = nowMs;
      updateScene();
      renderer.render(scene, camera);
      requestAnimationFrame(frame);
    }
    updateScene();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
})();
