/* Small STL viewer. Binary STL, millimetres, one finger to orbit, pinch to zoom. */
(function () {
  function parseSTL(buffer) {
    const view = new DataView(buffer);
    const count = view.getUint32(80, true);
    const vertices = new Float32Array(count * 9);
    let offset = 84;
    for (let i = 0; i < count; i++) {
      offset += 12; // normal
      for (let v = 0; v < 9; v++) {
        vertices[i * 9 + v] = view.getFloat32(offset, true);
        offset += 4;
      }
      offset += 2;
    }
    return vertices;
  }

  function Viewer(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) throw new Error("WebGL is not available in this browser.");
    this.gl = gl;
    this.yaw = 0.7;
    this.pitch = 0.45;
    this.dist = 1.75;
    this.count = 0;
    this.center = [0, 0, 0];
    this.radius = 1;
    const vs = `
      attribute vec3 aPos; attribute vec3 aN;
      uniform mat4 uMvp; uniform mat3 uN;
      uniform vec3 uCenter; uniform float uScale;
      varying vec3 vN;
      void main() {
        vN = normalize(uN * aN);
        vec3 p = (aPos - uCenter) * uScale;
        gl_Position = uMvp * vec4(p, 1.0);
      }`;
    const fs = `
      precision mediump float; varying vec3 vN;
      void main() {
        vec3 L = normalize(vec3(0.4, 0.6, 0.8));
        float d = abs(dot(normalize(vN), L));
        vec3 base = vec3(0.72, 0.58, 0.42);
        gl_FragColor = vec4(base * (0.35 + 0.75 * d), 1.0);
      }`;
    function shader(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    }
    const prog = gl.createProgram();
    gl.attachShader(prog, shader(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    this.prog = prog;
    this.buf = gl.createBuffer();
    this._bindPointer();
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.85, 0.82, 0.76, 1);
    this._pointers = new Map();
    this.onPick = null;
    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      this._pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, ox: e.clientX, oy: e.clientY });
    });
    canvas.addEventListener("pointerup", (e) => {
      const start = this._pointers.get(e.pointerId);
      this._pointers.delete(e.pointerId);
      if (start && this._pointers.size === 0 && Math.hypot(e.clientX - start.ox, e.clientY - start.oy) < 8 && this.onPick) {
        const hit = this.pick(e.clientX, e.clientY);
        if (hit) this.onPick(hit);
      }
    });
    canvas.addEventListener("pointercancel", (e) => this._pointers.delete(e.pointerId));
    canvas.addEventListener("pointermove", (e) => this._move(e));
    this._pick = null;
  }

  Viewer.prototype._bindPointer = function () {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.useProgram(this.prog);
    const pos = gl.getAttribLocation(this.prog, "aPos");
    const nrm = gl.getAttribLocation(this.prog, "aN");
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(nrm);
    gl.vertexAttribPointer(nrm, 3, gl.FLOAT, false, 24, 12);
  };

  Viewer.prototype.load = function (buffer) {
    const verts = parseSTL(buffer);
    const packed = new Float32Array(verts.length * 2);
    let min = [Infinity, Infinity, Infinity];
    let max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < verts.length; i += 9) {
      const ax = verts[i + 3] - verts[i], ay = verts[i + 4] - verts[i + 1], az = verts[i + 5] - verts[i + 2];
      const bx = verts[i + 6] - verts[i], by = verts[i + 7] - verts[i + 1], bz = verts[i + 8] - verts[i + 2];
      let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const len = Math.hypot(nx, ny, nz) || 1;
      nx /= len; ny /= len; nz /= len;
      for (let v = 0; v < 3; v++) {
        const x = verts[i + v * 3], y = verts[i + v * 3 + 1], z = verts[i + v * 3 + 2];
        const o = (i / 3 + v) * 6;
        packed[o] = x; packed[o + 1] = y; packed[o + 2] = z;
        packed[o + 3] = nx; packed[o + 4] = ny; packed[o + 5] = nz;
        min[0] = Math.min(min[0], x); min[1] = Math.min(min[1], y); min[2] = Math.min(min[2], z);
        max[0] = Math.max(max[0], x); max[1] = Math.max(max[1], y); max[2] = Math.max(max[2], z);
      }
    }
    this.positions = verts;
    this.center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    this.radius = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2 || 1;
    this.count = verts.length / 3;
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buf);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, packed, this.gl.STATIC_DRAW);
    this.draw();
  };

  Viewer.prototype._move = function (e) {
    const prev = this._pointers.get(e.pointerId);
    if (!prev) return;
    const pts = [...this._pointers.values()];
    if (this._pointers.size === 1) {
      this.yaw += (e.clientX - prev.x) * 0.01;
      this.pitch += (e.clientY - prev.y) * 0.01;
      this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch));
    } else if (pts.length >= 2) {
      const other = pts.find((p) => p !== prev) || pts[0];
      const before = Math.hypot(prev.x - other.x, prev.y - other.y);
      prev.x = e.clientX; prev.y = e.clientY;
      const after = Math.hypot(e.clientX - other.x, e.clientY - other.y);
      if (before > 0) this.dist *= before / after;
      this.dist = Math.max(1.2, Math.min(12, this.dist));
      this.draw();
      return;
    }
    prev.x = e.clientX; prev.y = e.clientY;
    this.draw();
  };

  function mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return o;
  }
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  }
  function look(eye, target, up) {
    const z = norm(sub(eye, target));
    const x = norm(cross(up, z));
    const y = cross(z, x);
    // Column-major, matching WebGL's uniformMatrix4fv.
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
    ]);
  }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

  Viewer.prototype.draw = function () {
    const gl = this.gl;
    const w = this.canvas.clientWidth || 300;
    const h = this.canvas.clientHeight || 300;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w * devicePixelRatio;
      this.canvas.height = h * devicePixelRatio;
    }
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.count) return;
    // The mesh stays in millimetres. The shader subtracts uCenter and multiplies
    // by uScale so the camera can sit a fixed distance from the origin.
    const eye = [
      Math.cos(this.pitch) * Math.sin(this.yaw) * this.dist,
      Math.sin(this.pitch) * this.dist,
      Math.cos(this.pitch) * Math.cos(this.yaw) * this.dist,
    ];
    const view = look(eye, [0, 0, 0], [0, 1, 0]);
    const proj = perspective(0.9, this.canvas.width / this.canvas.height, 0.05, 40);
    this._eye = eye;
    const mvp = mul(proj, view);
    gl.useProgram(this.prog);
    this._bindPointer();
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, "uMvp"), false, mvp);
    gl.uniform3f(gl.getUniformLocation(this.prog, "uCenter"), this.center[0], this.center[1], this.center[2]);
    gl.uniform1f(gl.getUniformLocation(this.prog, "uScale"), 1 / this.radius);
    const nmat = new Float32Array([view[0], view[1], view[2], view[4], view[5], view[6], view[8], view[9], view[10]]);
    gl.uniformMatrix3fv(gl.getUniformLocation(this.prog, "uN"), false, nmat);
    gl.drawArrays(gl.TRIANGLES, 0, this.count);
  };

  Viewer.prototype.pick = function (clientX, clientY) {
    if (!this.positions || !this._eye) return null;
    const rect = this.canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;
    const tan = Math.tan(0.45);
    const aspect = rect.width / rect.height;
    const forward = norm([-this._eye[0], -this._eye[1], -this._eye[2]]);
    const right = norm(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const dir = norm([
      forward[0] + right[0] * ndcX * tan * aspect + up[0] * ndcY * tan,
      forward[1] + right[1] * ndcX * tan * aspect + up[1] * ndcY * tan,
      forward[2] + right[2] * ndcX * tan * aspect + up[2] * ndcY * tan,
    ]);
    // Ray is in the normalized model space the shader uses. Convert the hit back to millimetres.
    const origin = this._eye;
    const scale = 1 / this.radius;
    const toView = (x, y, z) => [
      (x - this.center[0]) * scale,
      (y - this.center[1]) * scale,
      (z - this.center[2]) * scale,
    ];
    let bestT = Infinity;
    let best = null;
    const p = this.positions;
    for (let i = 0; i < p.length; i += 9) {
      const hit = rayTriangle(origin, dir,
        toView(p[i], p[i + 1], p[i + 2]),
        toView(p[i + 3], p[i + 4], p[i + 5]),
        toView(p[i + 6], p[i + 7], p[i + 8]));
      if (hit && hit.t < bestT && hit.t > 0) {
        bestT = hit.t;
        best = [
          hit.point[0] / scale + this.center[0],
          hit.point[1] / scale + this.center[1],
          hit.point[2] / scale + this.center[2],
        ];
      }
    }
    return best;
  };

  function rayTriangle(origin, dir, a, b, c) {
    const e1 = sub(b, a), e2 = sub(c, a);
    const pvec = cross(dir, e2);
    const det = dot(e1, pvec);
    if (Math.abs(det) < 1e-8) return null;
    const inv = 1 / det;
    const tvec = sub(origin, a);
    const u = dot(tvec, pvec) * inv;
    if (u < 0 || u > 1) return null;
    const qvec = cross(tvec, e1);
    const v = dot(dir, qvec) * inv;
    if (v < 0 || u + v > 1) return null;
    const t = dot(e2, qvec) * inv;
    if (t < 0) return null;
    return { t, point: [origin[0] + dir[0] * t, origin[1] + dir[1] * t, origin[2] + dir[2] * t] };
  }

  window.TrueShapeViewer = Viewer;
})();
