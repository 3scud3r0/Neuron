//! NEURON — portabilidade nativa incremental, arquivo único, Rust/std.
//!
//! Requer Rust 1.82 ou superior; testes executados em Linux x86-64.
//! Compilar: rustc --edition=2021 -O rust/neuron.rs -o neuron
//! Testar:   rustc --edition=2021 --test rust/neuron.rs -o neuron-tests && ./neuron-tests
//! Executar: ./neuron                 (chat; /ajuda lista comandos)
//!           ./neuron --coverage     (cobertura e diferenças explícitas)
//!           ./neuron --help
//!
//! Referência: python/neuron.py 9.0.0, SHA256
//! 244fecfbd3b3a675e14c139e357ba6ae3806dfd17381683876b6639242ba84e3.
//! Este arquivo NÃO é uma tradução integral dos 17.281 linhas do Python.
//! Não incorpora nem invoca o interpretador Python. Implementa nativamente o
//! encoder/decoder recorrente com atenção e BPTT, TCP, replay, aprendizado online,
//! provas polinomiais limitadas, perfis, datasets textuais, evolução transacional,
//! histórico encadeado, inspeção HTTP e uma seleção do córtex matemático.
//! `--coverage` é a especificação da paridade atual. HF/Torch, Arrow/Parquet/PDF,
//! SQLite/NPZ, o catálogo completo OMNIMAT, JIT, desktop, WebSocket, federação e
//! alteração de fonte Rust compilado não estão implementados neste port.
//! A API matemática usa argumentos JSON; nomes, assinaturas e domínios
//! numéricos podem diferir da API Python. Consulte --catalog e --coverage.
//!
//! Pesos iniciais aleatórios NÃO oferecem fluência pré-treinada. Recorrência
//! latente é computação mensurável, não evidência de pensamento ou veracidade.
//! TCP protege restrições locais de primeira ordem e probes finitos; não promete
//! ausência de esquecimento. Constituição numérica não prova alinhamento moral.
//! O modo irrestrito altera plasticidade, nunca permissões do sistema operacional.
//! Histórico em JSONL + SHA256 detecta alteração local, mas não é assinatura nem
//! protege contra um atacante que reescreve toda a cadeia. Todos os modos têm
//! limites explícitos. Não executa comandos/dados como código nem acessa rede
//! externa automaticamente. Dados persistidos são legíveis, não criptografados.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, String>;
const VERSION: &str = "9.0.0-rust-port.1";
const MAX_FILE: usize = 32 * 1024 * 1024;
const EPS: f64 = 1e-12;
fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(message.into())
}
fn ioerr(e: io::Error) -> String {
    e.to_string()
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// JSON independente de crates. Limites de profundidade/tamanho; nenhum eval.
#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}
impl Json {
    fn parse(text: &str) -> Result<Self> {
        if text.len() > MAX_FILE {
            return err("JSON excede 32 MiB");
        }
        let mut p = JsonParser {
            bytes: text.as_bytes(),
            pos: 0,
        };
        let value = p.value(0)?;
        p.ws();
        if p.pos != p.bytes.len() {
            return err("Conteúdo após JSON");
        }
        Ok(value)
    }
    fn get(&self, key: &str) -> &Json {
        if let Self::Obj(o) = self {
            o.get(key).unwrap_or(&Json::Null)
        } else {
            &Json::Null
        }
    }
    fn string(&self) -> Result<&str> {
        if let Self::Str(s) = self {
            Ok(s)
        } else {
            err("String esperada")
        }
    }
    fn number(&self) -> Result<f64> {
        if let Self::Num(n) = self {
            Ok(*n)
        } else {
            err("Número esperado")
        }
    }
    fn usize(&self) -> Result<usize> {
        let x = self.number()?;
        if x < 0. || x.fract() != 0. || x > usize::MAX as f64 {
            return err("Inteiro não negativo esperado");
        }
        Ok(x as usize)
    }
    fn array(&self) -> Result<&[Json]> {
        if let Self::Arr(a) = self {
            Ok(a)
        } else {
            err("Array esperado")
        }
    }
    fn encode(&self) -> String {
        match self {
            Self::Null => "null".into(),
            Self::Bool(b) => b.to_string(),
            Self::Num(n) => {
                if n.is_finite() {
                    n.to_string()
                } else {
                    "null".into()
                }
            }
            Self::Str(s) => quote(s),
            Self::Arr(a) => format!(
                "[{}]",
                a.iter().map(Self::encode).collect::<Vec<_>>().join(",")
            ),
            Self::Obj(o) => format!(
                "{{{}}}",
                o.iter()
                    .map(|(k, v)| format!("{}:{}", quote(k), v.encode()))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
fn obj<const N: usize>(pairs: [(&str, Json); N]) -> Json {
    Json::Obj(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn js(s: impl Into<String>) -> Json {
    Json::Str(s.into())
}
fn jn(x: impl Into<f64>) -> Json {
    Json::Num(x.into())
}
fn ja(x: &[f64]) -> Json {
    Json::Arr(x.iter().copied().map(Json::Num).collect())
}
struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl JsonParser<'_> {
    fn ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\r' | b'\n')
        {
            self.pos += 1;
        }
    }
    fn consume(&mut self, b: u8) -> bool {
        self.ws();
        if self.bytes.get(self.pos) == Some(&b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn literal(&mut self, s: &[u8], v: Json) -> Result<Json> {
        if self.bytes.get(self.pos..self.pos + s.len()) == Some(s) {
            self.pos += s.len();
            Ok(v)
        } else {
            err("Literal JSON inválido")
        }
    }
    fn string(&mut self) -> Result<String> {
        if !self.consume(b'"') {
            return err("String JSON esperada");
        }
        let mut out = String::new();
        loop {
            let b = *self.bytes.get(self.pos).ok_or("String incompleta")?;
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.bytes.get(self.pos).ok_or("Escape incompleto")?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xd800..=0xdbff).contains(&cp) {
                                if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                    return err("Surrogate incompleto");
                                }
                                self.pos += 2;
                                let low = self.hex4()?;
                                if !(0xdc00..=0xdfff).contains(&low) {
                                    return err("Surrogate inválido");
                                }
                                cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                            }
                            out.push(char::from_u32(cp).ok_or("Unicode inválido")?);
                        }
                        _ => return err("Escape JSON inválido"),
                    }
                }
                0..=31 => return err("Controle em string JSON"),
                32..=127 => out.push(b as char),
                _ => {
                    self.pos -= 1;
                    let s = std::str::from_utf8(&self.bytes[self.pos..])
                        .map_err(|_| "UTF-8 inválido")?;
                    let c = s.chars().next().ok_or("UTF-8 vazio")?;
                    self.pos += c.len_utf8();
                    out.push(c);
                }
            }
        }
    }
    fn hex4(&mut self) -> Result<u32> {
        let s = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or("Escape Unicode incompleto")?;
        self.pos += 4;
        u32::from_str_radix(std::str::from_utf8(s).map_err(|_| "Hex inválido")?, 16)
            .map_err(|_| "Hex inválido".into())
    }
    fn value(&mut self, depth: usize) -> Result<Json> {
        if depth > 64 {
            return err("JSON excede profundidade 64");
        }
        self.ws();
        match self.bytes.get(self.pos).copied() {
            Some(b'n') => self.literal(b"null", Json::Null),
            Some(b't') => self.literal(b"true", Json::Bool(true)),
            Some(b'f') => self.literal(b"false", Json::Bool(false)),
            Some(b'"') => Ok(js(self.string()?)),
            Some(b'[') => {
                self.pos += 1;
                let mut a = Vec::new();
                if self.consume(b']') {
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value(depth + 1)?);
                    if self.consume(b']') {
                        break;
                    }
                    if !self.consume(b',') {
                        return err("Vírgula JSON esperada");
                    }
                }
                Ok(Json::Arr(a))
            }
            Some(b'{') => {
                self.pos += 1;
                let mut o = BTreeMap::new();
                if self.consume(b'}') {
                    return Ok(Json::Obj(o));
                }
                loop {
                    let key = self.string()?;
                    if !self.consume(b':') {
                        return err("Dois pontos esperados");
                    }
                    if o.insert(key, self.value(depth + 1)?).is_some() {
                        return err("Chave JSON duplicada");
                    }
                    if self.consume(b'}') {
                        break;
                    }
                    if !self.consume(b',') {
                        return err("Vírgula JSON esperada");
                    }
                }
                Ok(Json::Obj(o))
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.pos;
                if self.bytes[self.pos] == b'-' {
                    self.pos += 1;
                }
                match self.bytes.get(self.pos) {
                    Some(b'0') => self.pos += 1,
                    Some(b'1'..=b'9') => {
                        self.pos += 1;
                        while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                            self.pos += 1;
                        }
                    }
                    _ => return err("Número JSON inválido"),
                };
                if self.bytes.get(self.pos) == Some(&b'.') {
                    self.pos += 1;
                    let s = self.pos;
                    while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                        self.pos += 1;
                    }
                    if s == self.pos {
                        return err("Fração JSON inválida");
                    }
                }
                if self
                    .bytes
                    .get(self.pos)
                    .is_some_and(|b| *b == b'e' || *b == b'E')
                {
                    self.pos += 1;
                    if self
                        .bytes
                        .get(self.pos)
                        .is_some_and(|b| *b == b'+' || *b == b'-')
                    {
                        self.pos += 1;
                    }
                    let s = self.pos;
                    while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                        self.pos += 1;
                    }
                    if s == self.pos {
                        return err("Expoente JSON inválido");
                    }
                }
                let s = std::str::from_utf8(&self.bytes[start..self.pos])
                    .map_err(|_| "Número inválido")?;
                let n: f64 = s.parse().map_err(|_| "Número inválido")?;
                if !n.is_finite() {
                    return err("Número não finito");
                }
                Ok(jn(n))
            }
            _ => err("JSON inválido"),
        }
    }
}

// SHA-256 conforme FIPS 180-4, checksum de checkpoints/eventos.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut bytes = data.to_vec();
    let bits = (bytes.len() as u64).wrapping_mul(8);
    bytes.push(128);
    while bytes.len() % 64 != 56 {
        bytes.push(0);
    }
    bytes.extend(bits.to_be_bytes());
    for block in bytes.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(c.try_into().unwrap());
        }
        for i in 16..64 {
            let a = w[i - 15];
            let b = w[i - 2];
            w[i] = w[i - 16]
                .wrapping_add(a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3))
                .wrapping_add(w[i - 7])
                .wrapping_add(b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10));
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let t = hh
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ (!e & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let u = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t);
            d = c;
            c = b;
            b = a;
            a = t.wrapping_add(u);
        }
        for (i, x) in [a, b, c, d, e, f, g, hh].iter().enumerate() {
            h[i] = h[i].wrapping_add(*x);
        }
    }
    let mut out = [0u8; 32];
    for (i, x) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}
fn hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}
fn digest(data: &[u8]) -> String {
    hex(&sha256(data))
}
fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let f = File::open(path).map_err(ioerr)?;
    if f.metadata().map_err(ioerr)?.len() > MAX_FILE as u64 {
        return err("Arquivo excede 32 MiB");
    }
    let mut data = Vec::new();
    f.take(MAX_FILE as u64 + 1)
        .read_to_end(&mut data)
        .map_err(ioerr)?;
    if data.len() > MAX_FILE {
        return err("Arquivo excede limite");
    }
    Ok(data)
}
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(ioerr)?;
    let tmp = parent.join(format!(
        ".neuron-{}-{}.tmp",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let result = (|| {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(ioerr)?;
        f.write_all(bytes).map_err(ioerr)?;
        f.sync_all().map_err(ioerr)?;
        fs::rename(&tmp, path).map_err(ioerr)?;
        #[cfg(unix)]
        File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(ioerr)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

#[derive(Clone, Debug)]
struct Rng {
    state: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2. * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
    fn index(&mut self, n: usize) -> usize {
        let n = n as u64;
        let cut = n.wrapping_neg() % n;
        loop {
            let x = self.next();
            if x >= cut {
                return (x % n) as usize;
            }
        }
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn softmax(a: &[f64]) -> Vec<f64> {
    let m = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut v: Vec<_> = a.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = v.iter().sum();
    for x in &mut v {
        *x /= s;
    }
    v
}
fn matvec(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    debug_assert_eq!(w.len(), rows * cols);
    w.chunks_exact(cols).map(|r| dot(r, x)).collect()
}
fn transvec(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.; cols];
    for i in 0..rows {
        for j in 0..cols {
            y[j] += w[i * cols + j] * x[i];
        }
    }
    y
}
fn outer_add(w: &mut [f64], a: &[f64], b: &[f64]) {
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            w[i * b.len() + j] += x * y;
        }
    }
}
fn add(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

#[derive(Clone, Debug)]
struct Transport {
    delta: Vec<f64>,
    fallback: bool,
    iterations: usize,
    max_violation: f64,
}
fn transport_displacement(
    free: &[f64],
    memories: &[Vec<f64>],
    importance: &[f64],
    strength: f64,
) -> Result<Transport> {
    if free.len() != importance.len()
        || !strength.is_finite()
        || strength < 0.
        || free.iter().chain(importance).any(|x| !x.is_finite())
        || memories
            .iter()
            .any(|r| r.len() != free.len() || r.iter().any(|x| !x.is_finite()))
    {
        return err("TCP: dimensões/valores inválidos");
    }
    let rows: Vec<Vec<f64>> = memories
        .iter()
        .filter_map(|r| {
            let n = norm(r);
            if n > EPS {
                Some(r.iter().map(|x| x / n).collect())
            } else {
                None
            }
        })
        .collect();
    let mut result = Transport {
        delta: free.to_vec(),
        fallback: false,
        iterations: 0,
        max_violation: 0.,
    };
    if rows.is_empty() || rows.iter().all(|r| dot(r, free) <= 1e-9) {
        return Ok(result);
    }
    let salience: Vec<_> = importance.iter().map(|x| x.max(0.).sqrt()).collect();
    let mean = salience.iter().sum::<f64>() / (salience.len().max(1) as f64);
    let inv: Vec<_> = salience
        .iter()
        .map(|s| 1. / (1. + strength * s / (mean + s + EPS)))
        .collect();
    let n = rows.len();
    let mut h = vec![vec![0.; n]; n];
    let q: Vec<_> = rows.iter().map(|r| dot(r, free)).collect();
    for i in 0..n {
        for j in 0..n {
            h[i][j] = rows[i]
                .iter()
                .zip(&rows[j])
                .zip(&inv)
                .map(|((a, b), m)| a * b * m)
                .sum();
        }
    }
    let mut lambda = vec![0.; n];
    for sweep in 0..96 {
        let mut change: f64 = 0.;
        for i in 0..n {
            let next = (lambda[i] + (q[i] - dot(&h[i], &lambda)) / h[i][i].max(EPS)).max(0.);
            change = change.max((next - lambda[i]).abs());
            lambda[i] = next;
        }
        result.iterations = sweep + 1;
        if change < 1e-9 {
            break;
        }
    }
    for j in 0..free.len() {
        result.delta[j] -= inv[j] * rows.iter().zip(&lambda).map(|(r, l)| r[j] * l).sum::<f64>();
    }
    result.max_violation = rows
        .iter()
        .map(|r| dot(r, &result.delta))
        .fold(0., f64::max);
    result.fallback = result.max_violation > 1e-8 || result.delta.iter().any(|x| !x.is_finite());
    if result.fallback {
        result.delta.fill(0.);
    }
    Ok(result)
}
fn constitutional_rows(n: usize) -> Vec<Vec<f64>> {
    if n == 0 {
        return vec![];
    }
    let scale = (n as f64).sqrt();
    let a = vec![1. / scale; n];
    let mut b: Vec<_> = (0..n)
        .map(|i| if i % 2 == 0 { 1. / scale } else { -1. / scale })
        .collect();
    let prod = dot(&a, &b);
    for (x, y) in b.iter_mut().zip(&a) {
        *x -= prod * y;
    }
    let bnorm = norm(&b);
    let mut rows = vec![a];
    if bnorm > EPS {
        for x in &mut b {
            *x /= bnorm;
        }
        rows.push(b);
    }
    rows
}

#[derive(Clone, Debug)]
struct Reservoir<T> {
    items: Vec<T>,
    seen: u64,
    capacity: usize,
    rng: Rng,
}
impl<T: Clone> Reservoir<T> {
    fn new(capacity: usize, seed: u64) -> Self {
        Self {
            items: Vec::new(),
            seen: 0,
            capacity,
            rng: Rng::new(seed),
        }
    }
    fn push(&mut self, item: T) {
        self.seen = self.seen.saturating_add(1);
        if self.items.len() < self.capacity {
            self.items.push(item);
        } else if self.capacity > 0 {
            let j = self.rng.next() % self.seen;
            if j < self.capacity as u64 {
                self.items[j as usize] = item;
            }
        }
    }
}
#[derive(Clone, Debug)]
struct PageHinkley {
    count: u64,
    mean: f64,
    cumulative: f64,
    minimum: f64,
    threshold: f64,
}
impl PageHinkley {
    fn new(threshold: f64) -> Self {
        Self {
            count: 0,
            mean: 0.,
            cumulative: 0.,
            minimum: 0.,
            threshold,
        }
    }
    fn update(&mut self, x: f64) -> bool {
        if !x.is_finite() {
            return false;
        }
        let x = x.clamp(0., 5.);
        self.count += 1;
        self.mean += (x - self.mean) / self.count as f64;
        self.cumulative += x - self.mean - 0.005;
        self.minimum = self.minimum.min(self.cumulative);
        let drift = self.count >= 20 && self.cumulative - self.minimum > self.threshold;
        if drift {
            *self = Self::new(self.threshold);
        }
        drift
    }
}

type Params = BTreeMap<String, Vec<f64>>;
fn zero_params(p: &Params) -> Params {
    p.iter()
        .map(|(k, v)| (k.clone(), vec![0.; v.len()]))
        .collect()
}
fn flatten(p: &Params) -> Vec<f64> {
    p.values().flatten().copied().collect()
}
fn apply_flat(p: &mut Params, d: &[f64]) -> Result<()> {
    if p.values().map(Vec::len).sum::<usize>() != d.len() {
        return err("Delta incompatível");
    }
    let mut i = 0;
    for a in p.values_mut() {
        for x in a {
            *x += d[i];
            i += 1;
        }
    }
    Ok(())
}
fn params_json(p: &Params) -> Json {
    Json::Obj(p.iter().map(|(k, v)| (k.clone(), ja(v))).collect())
}
fn params_from_json(j: &Json) -> Result<Params> {
    if let Json::Obj(o) = j {
        o.iter()
            .map(|(k, v)| {
                Ok((
                    k.clone(),
                    v.array()?
                        .iter()
                        .map(Json::number)
                        .collect::<Result<Vec<_>>>()?,
                ))
            })
            .collect()
    } else {
        err("Parâmetros esperados")
    }
}

// Tradução direta das equações do DendriticCore Python: mistura de especialistas,
// portões sigmoid, ramos tanh + seno e responsabilidades em log-space.
#[derive(Clone, Debug)]
struct DendriticCore {
    features: usize,
    branches: usize,
    outputs: usize,
    active: usize,
    max_experts: usize,
    p: Params,
}
struct DendriticCache {
    a: Vec<f64>,
    gate: Vec<f64>,
    wave: Vec<f64>,
    hidden: Vec<f64>,
    z: Vec<f64>,
    pi: Vec<f64>,
    log_pi: Vec<f64>,
    prob: Vec<f64>,
    output: Vec<f64>,
}
impl DendriticCore {
    fn new(
        features: usize,
        branches: usize,
        outputs: usize,
        max_experts: usize,
        seed: u64,
    ) -> Result<Self> {
        if features == 0
            || features > 1024
            || branches == 0
            || branches > 128
            || outputs < 2
            || outputs > 1024
            || max_experts == 0
            || max_experts > 16
        {
            return err("Dimensões dendríticas inválidas");
        }
        let mut rng = Rng::new(seed);
        let f = features;
        let b = branches;
        let e = max_experts;
        let o = outputs;
        let mut p = Params::new();
        for (name, size, scale) in [
            ("W", e * b * f, 1. / (f as f64).sqrt()),
            ("V", e * b * f, 0.3 / (f as f64).sqrt()),
            ("r", e * b, 0.1),
            ("O", e * o * b, 0.25 / (f as f64).sqrt()),
            ("R", e * f, 0.05 / (f as f64).sqrt()),
        ] {
            p.insert(
                name.into(),
                (0..size).map(|_| rng.normal() * scale).collect(),
            );
        }
        for (name, size, value) in [
            ("b", e * b, 0.),
            ("c", e * b, 1.),
            ("S", e * o * f, 0.),
            ("d", e * o, 0.),
            ("beta", e, 0.),
        ] {
            p.insert(name.into(), vec![value; size]);
        }
        Ok(Self {
            features,
            branches,
            outputs,
            active: 1,
            max_experts,
            p,
        })
    }
    fn forward(&self, u: &[f64]) -> Result<DendriticCache> {
        if u.len() != self.features || u.iter().any(|x| !x.is_finite()) {
            return err("Entrada dendrítica inválida");
        }
        let (f, b, o, e) = (self.features, self.branches, self.outputs, self.active);
        let p = &self.p;
        let mut a = vec![0.; e * b];
        let mut gate = a.clone();
        let mut wave = a.clone();
        let mut hidden = a.clone();
        let mut z = vec![0.; e * o];
        let mut route = vec![0.; e];
        let mut prob = z.clone();
        for i in 0..e {
            for j in 0..b {
                let k = i * b + j;
                a[k] = dot(&p["W"][k * f..(k + 1) * f], u) + p["b"][k];
                let q = dot(&p["V"][k * f..(k + 1) * f], u) + p["c"][k];
                gate[k] = 1. / (1. + (-q).exp());
                wave[k] = a[k].tanh() + p["r"][k].tanh() * a[k].sin();
                hidden[k] = gate[k] * wave[k];
            }
            for j in 0..o {
                let k = i * o + j;
                z[k] = dot(&p["O"][k * b..(k + 1) * b], &hidden[i * b..(i + 1) * b])
                    / (b as f64).sqrt()
                    + dot(&p["S"][k * f..(k + 1) * f], u)
                    + p["d"][k];
            }
            prob[i * o..(i + 1) * o].copy_from_slice(&softmax(&z[i * o..(i + 1) * o]));
            route[i] = dot(&p["R"][i * f..(i + 1) * f], u) + p["beta"][i];
        }
        let pi = softmax(&route);
        let route_max = route.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_norm = route_max
            + route
                .iter()
                .map(|x| (x - route_max).exp())
                .sum::<f64>()
                .ln();
        let log_pi = route.iter().map(|x| x - log_norm).collect();
        let mut output = vec![0.; o];
        for i in 0..e {
            for j in 0..o {
                output[j] += pi[i] * prob[i * o + j];
            }
        }
        Ok(DendriticCache {
            a,
            gate,
            wave,
            hidden,
            z,
            pi,
            log_pi,
            prob,
            output,
        })
    }
    fn gradients(&self, u: &[f64], target: usize) -> Result<(f64, Params)> {
        if target >= self.outputs {
            return err("Classe inválida");
        }
        let q = self.forward(u)?;
        let (f, b, o, e) = (self.features, self.branches, self.outputs, self.active);
        let p = &self.p;
        let mut g = zero_params(p);
        // Log-sum-exp mantém responsabilidades mesmo com probabilidades subnormais.
        let logsum = |x: &[f64]| {
            let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
        };
        let joint: Vec<_> = (0..e)
            .map(|i| q.log_pi[i] + q.z[i * o + target] - logsum(&q.z[i * o..(i + 1) * o]))
            .collect();
        let logoutput = logsum(&joint);
        for i in 0..e {
            let mass = (joint[i] - logoutput).exp();
            let dr = q.pi[i] - mass;
            g.get_mut("beta").unwrap()[i] = dr;
            for j in 0..f {
                g.get_mut("R").unwrap()[i * f + j] = dr * u[j];
            }
            let mut dh = vec![0.; b];
            for j in 0..o {
                let k = i * o + j;
                let dz = mass * (q.prob[k] - if j == target { 1. } else { 0. });
                g.get_mut("d").unwrap()[k] = dz;
                for l in 0..f {
                    g.get_mut("S").unwrap()[k * f + l] = dz * u[l];
                }
                for l in 0..b {
                    g.get_mut("O").unwrap()[k * b + l] =
                        dz * q.hidden[i * b + l] / (b as f64).sqrt();
                    dh[l] += dz * p["O"][k * b + l] / (b as f64).sqrt();
                }
            }
            for (j, &dhj) in dh.iter().enumerate() {
                let k = i * b + j;
                let ta = q.a[k].tanh();
                let amp = p["r"][k].tanh();
                let da = dhj * q.gate[k] * (1. - ta * ta + amp * q.a[k].cos());
                let dg = dhj * q.wave[k] * q.gate[k] * (1. - q.gate[k]);
                g.get_mut("b").unwrap()[k] = da;
                g.get_mut("c").unwrap()[k] = dg;
                g.get_mut("r").unwrap()[k] = dhj * q.gate[k] * q.a[k].sin() * (1. - amp * amp);
                for l in 0..f {
                    g.get_mut("W").unwrap()[k * f + l] = da * u[l];
                    g.get_mut("V").unwrap()[k * f + l] = dg * u[l];
                }
            }
        }
        Ok((-logoutput, g))
    }
    fn grow(&mut self, source: usize) -> Result<bool> {
        if source >= self.active {
            return err("Especialista inválido");
        }
        if self.active == self.max_experts {
            return Ok(false);
        }
        let idx = self.active;
        for a in self.p.values_mut() {
            let stride = a.len() / self.max_experts;
            let from = a[source * stride..(source + 1) * stride].to_vec();
            a[idx * stride..(idx + 1) * stride].copy_from_slice(&from);
        }
        let beta = self.p.get_mut("beta").unwrap();
        beta[source] -= 2f64.ln();
        beta[idx] = beta[source];
        self.active += 1;
        Ok(true)
    }
}

#[derive(Clone, Debug)]
struct SeqConfig {
    input_tokens: usize,
    output_tokens: usize,
    thought_budget: usize,
    inference_seconds: f64,
}
impl Default for SeqConfig {
    fn default() -> Self {
        Self {
            input_tokens: 128,
            output_tokens: 64,
            thought_budget: 0,
            inference_seconds: 10.,
        }
    }
}
impl SeqConfig {
    fn validate(&self) -> Result<()> {
        if self.input_tokens == 0
            || self.input_tokens > 1024
            || self.output_tokens < 2
            || self.output_tokens > 512
            || self.thought_budget > 256
            || self.thought_budget * self.output_tokens > 8192
            || !self.inference_seconds.is_finite()
            || !(0.1..=30.).contains(&self.inference_seconds)
        {
            return err("Limites: entrada 1..1024, saída 2..512, esforço 0..256, total latente <=8192, prazo .1..30 s");
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
struct NativeSeq2Seq {
    dimension: usize,
    vocabulary: Vec<String>,
    lookup: BTreeMap<char, usize>,
    p: Params,
    config: SeqConfig,
    updates: u64,
}
#[derive(Clone, Debug)]
struct DecoderRecord {
    prev: usize,
    before: Vec<f64>,
    hidden: Vec<f64>,
    weights: Vec<f64>,
    attended: Vec<f64>,
    prob: Vec<f64>,
    token: usize,
    latents: Vec<Vec<f64>>,
}
struct Forward {
    loss: f64,
    states: Vec<Vec<f64>>,
    records: Vec<DecoderRecord>,
    tokens: Vec<usize>,
}
impl NativeSeq2Seq {
    fn new(dimension: usize, seed: u64) -> Result<Self> {
        if !(8..=128).contains(&dimension) {
            return err("Dimensão entre 8 e 128");
        }
        let mut vocabulary = vec!["<BOS>".into(), "<EOS>".into(), "<UNK>".into()];
        let mut seen = BTreeSet::new();
        for c in " \n"
            .chars()
            .chain((33u8..127).map(char::from))
            .chain("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ—…“”".chars())
        {
            if seen.insert(c) {
                vocabulary.push(c.to_string());
            }
        }
        let lookup = vocabulary
            .iter()
            .enumerate()
            .skip(3)
            .map(|(i, c)| (c.chars().next().unwrap(), i))
            .collect();
        let v = vocabulary.len();
        let h = dimension;
        let mut rng = Rng::new(seed);
        let mut p = Params::new();
        for (name, size) in [
            ("E", v * h),
            ("A", h * h),
            ("D", h * h),
            ("C", h * h),
            ("O", v * h),
            ("b", v),
            ("B", h * 48),
        ] {
            let scale = if name == "B" {
                0.015
            } else if name == "b" {
                0.
            } else {
                0.15
            };
            p.insert(
                name.into(),
                (0..size).map(|_| rng.normal() * scale).collect(),
            );
        }
        Ok(Self {
            dimension,
            vocabulary,
            lookup,
            p,
            config: SeqConfig::default(),
            updates: 0,
        })
    }
    fn tokens(&self, text: &str, limit: usize) -> Vec<usize> {
        let chars: Vec<_> = text.chars().collect();
        let mut ids: Vec<_> = chars[chars.len().saturating_sub(limit)..]
            .iter()
            .map(|c| *self.lookup.get(c).unwrap_or(&2))
            .collect();
        if ids.is_empty() {
            ids.push(0);
        }
        ids
    }
    fn encoder(&self, ids: &[usize]) -> Vec<Vec<f64>> {
        let h = self.dimension;
        let mut state = vec![0.; h];
        let mut states = Vec::with_capacity(ids.len());
        for &id in ids {
            let mut next = matvec(&self.p["A"], h, h, &state);
            for (j, x) in next.iter_mut().enumerate() {
                *x = (*x + self.p["E"][id * h + j]).tanh();
            }
            states.push(next.clone());
            state = next;
        }
        states
    }
    fn forward(&self, text: &str, context: &[f64], targets: Option<&[usize]>) -> Result<Forward> {
        self.config.validate()?;
        if context.len() != 48 || context.iter().any(|x| !x.is_finite()) {
            return err("Contexto deve ter 48 números finitos");
        }
        let p = &self.p;
        let h = self.dimension;
        let v = self.vocabulary.len();
        let states = self.encoder(&self.tokens(text, self.config.input_tokens));
        let mut hidden = states.last().unwrap().clone();
        let mut prev = 0;
        let mut records = Vec::new();
        let mut tokens = Vec::new();
        let mut loss = 0.;
        let count = targets.map_or(self.config.output_tokens, <[usize]>::len);
        let deadline = Instant::now() + Duration::from_secs_f64(self.config.inference_seconds);
        if self.config.thought_budget * count > 8192 {
            return err("Orçamento latente excedido");
        }
        for t in 0..count {
            if Instant::now() > deadline {
                return err("Prazo de computação excedido");
            }
            let before = hidden.clone();
            let scores: Vec<_> = states
                .iter()
                .map(|s| dot(s, &before) / (h as f64).sqrt())
                .collect();
            let weights = softmax(&scores);
            let mut attended = vec![0.; h];
            for (w, s) in weights.iter().zip(&states) {
                for j in 0..h {
                    attended[j] += w * s[j];
                }
            }
            let mut base = matvec(&p["D"], h, h, &hidden);
            add(&mut base, &matvec(&p["C"], h, h, &attended));
            add(&mut base, &matvec(&p["B"], h, 48, context));
            for (j, x) in base.iter_mut().enumerate() {
                *x = (*x + p["E"][prev * h + j]).tanh();
            }
            hidden = base.clone();
            let mut latents = vec![base];
            for _ in 0..self.config.thought_budget {
                if Instant::now() > deadline {
                    return err("Prazo latente excedido");
                }
                hidden = matvec(&p["D"], h, h, &hidden)
                    .iter()
                    .map(|x| x.tanh())
                    .collect();
                latents.push(hidden.clone());
            }
            let mut logits = matvec(&p["O"], v, h, &hidden);
            add(&mut logits, &p["b"]);
            let prob = softmax(&logits);
            let token = if let Some(targets) = targets {
                targets[t]
            } else {
                (1..v).max_by(|&a, &b| prob[a].total_cmp(&prob[b])).unwrap()
            };
            if token >= v {
                return err("Token alvo inválido");
            }
            loss -= prob[token].max(1e-300).ln();
            tokens.push(token);
            records.push(DecoderRecord {
                prev,
                before,
                hidden: hidden.clone(),
                weights,
                attended,
                prob,
                token,
                latents,
            });
            prev = token;
            if targets.is_none() && token == 1 {
                break;
            }
        }
        Ok(Forward {
            loss: loss / count.max(1) as f64,
            states,
            records,
            tokens,
        })
    }
    fn loss_grad(&self, text: &str, context: &[f64], answer: &str) -> Result<(f64, Params)> {
        let mut targets = self.tokens(answer, self.config.output_tokens - 1);
        targets.push(1);
        let f = self.forward(text, context, Some(&targets))?;
        let p = &self.p;
        let h = self.dimension;
        let v = self.vocabulary.len();
        let mut g = zero_params(p);
        let mut ds = vec![vec![0.; h]; f.states.len()];
        let mut dh = vec![0.; h];
        for r in f.records.iter().rev() {
            let mut dl = r.prob.clone();
            dl[r.token] -= 1.;
            for x in &mut dl {
                *x /= f.records.len() as f64;
            }
            outer_add(g.get_mut("O").unwrap(), &dl, &r.hidden);
            add(g.get_mut("b").unwrap(), &dl);
            add(&mut dh, &transvec(&p["O"], v, h, &dl));
            for i in (1..r.latents.len()).rev() {
                let dt: Vec<_> = dh
                    .iter()
                    .zip(&r.latents[i])
                    .map(|(d, x)| d * (1. - x * x))
                    .collect();
                outer_add(g.get_mut("D").unwrap(), &dt, &r.latents[i - 1]);
                dh = transvec(&p["D"], h, h, &dt);
            }
            let dt: Vec<_> = dh
                .iter()
                .zip(&r.latents[0])
                .map(|(d, x)| d * (1. - x * x))
                .collect();
            add(
                &mut g.get_mut("E").unwrap()[r.prev * h..(r.prev + 1) * h],
                &dt,
            );
            outer_add(g.get_mut("D").unwrap(), &dt, &r.before);
            outer_add(g.get_mut("C").unwrap(), &dt, &r.attended);
            outer_add(g.get_mut("B").unwrap(), &dt, context);
            let dc = transvec(&p["C"], h, h, &dt);
            let dw: Vec<_> = f.states.iter().map(|s| dot(s, &dc)).collect();
            let dwmean = dot(&dw, &r.weights);
            let de: Vec<_> = dw
                .iter()
                .zip(&r.weights)
                .map(|(d, w)| w * (d - dwmean) / (h as f64).sqrt())
                .collect();
            for i in 0..ds.len() {
                for (j, &dcj) in dc.iter().enumerate() {
                    ds[i][j] += r.weights[i] * dcj + de[i] * r.before[j];
                }
            }
            dh = transvec(&p["D"], h, h, &dt);
            for (i, s) in f.states.iter().enumerate() {
                for j in 0..h {
                    dh[j] += s[j] * de[i];
                }
            }
        }
        add(ds.last_mut().unwrap(), &dh);
        dh.fill(0.);
        let ids = self.tokens(text, self.config.input_tokens);
        let zero = vec![0.; h];
        for t in (0..f.states.len()).rev() {
            let dt: Vec<_> = (0..h)
                .map(|j| (ds[t][j] + dh[j]) * (1. - f.states[t][j] * f.states[t][j]))
                .collect();
            add(
                &mut g.get_mut("E").unwrap()[ids[t] * h..(ids[t] + 1) * h],
                &dt,
            );
            outer_add(
                g.get_mut("A").unwrap(),
                &dt,
                if t == 0 { &zero } else { &f.states[t - 1] },
            );
            dh = transvec(&p["A"], h, h, &dt);
        }
        if !f.loss.is_finite() || g.values().flatten().any(|x| !x.is_finite()) {
            return err("Gradiente não finito");
        }
        Ok((f.loss, g))
    }
    fn generate(&self, text: &str, context: &[f64]) -> Result<(String, Json)> {
        let f = self.forward(text, context, None)?;
        let output = f
            .tokens
            .iter()
            .filter_map(|&t| {
                if t > 2 {
                    Some(self.vocabulary[t].as_str())
                } else if t == 2 {
                    Some("?")
                } else {
                    None
                }
            })
            .collect::<String>()
            .trim()
            .to_string();
        let trace = obj([
            ("architecture", js("rust-native-rnn-encoder-decoder")),
            ("tokens", jn(f.tokens.len() as f64)),
            (
                "latent_iterations",
                jn((f.tokens.len() * self.config.thought_budget) as f64),
            ),
            ("thought_budget", jn(self.config.thought_budget as f64)),
            (
                "encoder_states",
                Json::Arr(f.states.iter().map(|v| ja(v)).collect()),
            ),
            (
                "decoder_states",
                Json::Arr(f.records.iter().map(|r| ja(&r.hidden)).collect()),
            ),
            (
                "cross_attention",
                Json::Arr(f.records.iter().map(|r| ja(&r.weights)).collect()),
            ),
            (
                "limitations",
                js("Pesos locais, sem fluência garantida; estados não são raciocínio textual."),
            ),
        ]);
        Ok((output, trace))
    }
    fn signature(&self) -> String {
        let mut bytes = Vec::new();
        bytes.extend((self.dimension as u64).to_le_bytes());
        for (k, a) in &self.p {
            bytes.extend(k.as_bytes());
            for x in a {
                bytes.extend(x.to_le_bytes());
            }
        }
        digest(&bytes)
    }
    fn to_json(&self) -> Json {
        obj([
            ("dimension", jn(self.dimension as f64)),
            ("parameters", params_json(&self.p)),
            ("updates", js(self.updates.to_string())),
            ("input_tokens", jn(self.config.input_tokens as f64)),
            ("output_tokens", jn(self.config.output_tokens as f64)),
            ("thought_budget", jn(self.config.thought_budget as f64)),
            ("inference_seconds", jn(self.config.inference_seconds)),
        ])
    }
    fn from_json(j: &Json) -> Result<Self> {
        let mut m = Self::new(j.get("dimension").usize()?, 0)?;
        let p = params_from_json(j.get("parameters"))?;
        if p.len() != m.p.len()
            || m.p.iter().any(|(k, v)| {
                p.get(k)
                    .is_none_or(|a| a.len() != v.len() || a.iter().any(|x| !x.is_finite()))
            })
        {
            return err("Checkpoint: esquema de pesos incompatível");
        }
        m.p = p;
        m.updates = j
            .get("updates")
            .string()?
            .parse()
            .map_err(|_| "Contador inválido")?;
        m.config = SeqConfig {
            input_tokens: j.get("input_tokens").usize()?,
            output_tokens: j.get("output_tokens").usize()?,
            thought_budget: j.get("thought_budget").usize()?,
            inference_seconds: j.get("inference_seconds").number()?,
        };
        m.config.validate()?;
        Ok(m)
    }
    // Ampliação funcional: neurônios novos começam desconectados da saída.
    // Atenção usa sqrt(H), portanto a migração exige gate empírico; não há
    // alegação de equivalência funcional para crescimento do Seq2Seq.
    fn resized(&self, new_h: usize) -> Result<Self> {
        let mut candidate = Self::new(new_h, 711)?;
        candidate.config = self.config.clone();
        candidate.updates = self.updates;
        let old_h = self.dimension;
        for name in ["A", "C", "D"] {
            candidate.p.get_mut(name).unwrap().fill(0.);
            for i in 0..old_h.min(new_h) {
                for j in 0..old_h.min(new_h) {
                    candidate.p.get_mut(name).unwrap()[i * new_h + j] = self.p[name][i * old_h + j];
                }
            }
        }
        for name in ["E", "O"] {
            candidate.p.get_mut(name).unwrap().fill(0.);
            for i in 0..self.vocabulary.len() {
                for j in 0..old_h.min(new_h) {
                    candidate.p.get_mut(name).unwrap()[i * new_h + j] = self.p[name][i * old_h + j];
                }
            }
        }
        candidate.p.get_mut("B").unwrap().fill(0.);
        for i in 0..old_h.min(new_h) {
            candidate.p.get_mut("B").unwrap()[i * 48..(i + 1) * 48]
                .copy_from_slice(&self.p["B"][i * 48..(i + 1) * 48]);
        }
        candidate.p.insert("b".into(), self.p["b"].clone());
        Ok(candidate)
    }
}

#[derive(Clone, Debug)]
struct Pair {
    input: String,
    output: String,
    trusted: bool,
}
impl Pair {
    fn json(&self) -> Json {
        obj([
            ("input", js(&self.input)),
            ("output", js(&self.output)),
            ("trusted", Json::Bool(self.trusted)),
        ])
    }
    fn from_json(j: &Json) -> Result<Self> {
        let input = j.get("input").string()?.to_string();
        let output = j.get("output").string()?.to_string();
        if input.len() > 16384 || output.len() > 16384 {
            return err("Exemplo excede 16 KiB por campo");
        }
        Ok(Self {
            input,
            output,
            trusted: j.get("trusted") == &Json::Bool(true),
        })
    }
}
fn zero_context() -> [f64; 48] {
    [0.; 48]
}
#[derive(Clone, Debug)]
struct TrainReport {
    accepted: bool,
    loss_before: f64,
    loss_after: f64,
    delta_l2: f64,
    fallback: bool,
    retention_worst: f64,
    reason: String,
}
impl TrainReport {
    fn json(&self) -> Json {
        obj([
            ("accepted", Json::Bool(self.accepted)),
            ("loss_before", jn(self.loss_before)),
            ("loss_after", jn(self.loss_after)),
            ("delta_l2", jn(self.delta_l2)),
            ("fallback", Json::Bool(self.fallback)),
            ("retention_worst", jn(self.retention_worst)),
            ("reason", js(&self.reason)),
        ])
    }
}
fn train_pair(
    model: &mut NativeSeq2Seq,
    pair: &Pair,
    replay: &[Pair],
    restricted: bool,
    rate: f64,
) -> Result<TrainReport> {
    if !rate.is_finite() || rate <= 0. || rate > 0.5 {
        return err("Taxa deve estar em (0,.5]");
    }
    let context = zero_context();
    let (loss_before, gradient) = model.loss_grad(&pair.input, &context, &pair.output)?;
    let flat = flatten(&gradient);
    let scale = rate / (norm(&flat) / 2.).max(1.);
    let free: Vec<_> = flat.iter().map(|x| -scale * x).collect();
    let mut memories = Vec::new();
    let mut before_losses = Vec::new();
    for p in replay.iter().rev().take(4) {
        let (l, g) = model.loss_grad(&p.input, &context, &p.output)?;
        memories.push(flatten(&g));
        before_losses.push((p, l));
    }
    if restricted {
        for r in constitutional_rows(flat.len()) {
            memories.push(r.iter().map(|x| -x).collect());
            memories.push(r);
        }
    }
    let projected = transport_displacement(&free, &memories, &vec![0.; flat.len()], 2.)?;
    let mut candidate = model.clone();
    apply_flat(&mut candidate.p, &projected.delta)?;
    let actual: Vec<_> = flatten(&candidate.p)
        .iter()
        .zip(flatten(&model.p))
        .map(|(a, b)| a - b)
        .collect();
    let mut accepted = actual.iter().all(|x| x.is_finite())
        && memories
            .iter()
            .all(|g| dot(g, &actual) <= 1e-7 * norm(g).max(1.));
    let loss_after = candidate.loss_grad(&pair.input, &context, &pair.output)?.0;
    accepted &= loss_after <= loss_before + 1e-9;
    let mut retention_worst: f64 = 0.;
    for (p, l) in before_losses {
        let after = candidate.loss_grad(&p.input, &context, &p.output)?.0;
        retention_worst = retention_worst.max(after - l);
    }
    accepted &= retention_worst <= 0.02;
    let changed = norm(&actual) > 1e-14;
    accepted &= changed;
    let reason = if accepted {
        "accepted"
    } else if !changed {
        "zero_step"
    } else if retention_worst > 0.02 {
        "retention_gate"
    } else {
        "loss_or_constraint_gate"
    }
    .into();
    if accepted {
        candidate.updates += 1;
        *model = candidate;
    }
    Ok(TrainReport {
        accepted,
        loss_before,
        loss_after,
        delta_l2: if accepted { norm(&actual) } else { 0. },
        fallback: projected.fallback,
        retention_worst,
        reason,
    })
}

// Álgebra exata sobre Q: AST restrita a polinômios, sem funções ou avaliação
// arbitrária. Inteiros i128 com overflow verificado; não simula inteiros ilimitados.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rational {
    n: i128,
    d: i128,
}
fn gcd128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
impl Rational {
    fn new(n: i128, d: i128) -> Result<Self> {
        if d == 0 {
            return err("Divisão por zero");
        }
        let sign = if d < 0 { -1 } else { 1 };
        let n = n.checked_mul(sign).ok_or("Overflow racional")?;
        let d = d.checked_mul(sign).ok_or("Overflow racional")?;
        let g = gcd128(n.unsigned_abs(), d as u128);
        Ok(Self {
            n: n / (g as i128),
            d: d / (g as i128),
        })
    }
    fn integer(n: i128) -> Self {
        Self { n, d: 1 }
    }
    fn add(self, b: Self) -> Result<Self> {
        let n = self
            .n
            .checked_mul(b.d)
            .and_then(|x| b.n.checked_mul(self.d).and_then(|y| x.checked_add(y)))
            .ok_or("Overflow racional")?;
        Self::new(n, self.d.checked_mul(b.d).ok_or("Overflow racional")?)
    }
    fn mul(self, b: Self) -> Result<Self> {
        Self::new(
            self.n.checked_mul(b.n).ok_or("Overflow racional")?,
            self.d.checked_mul(b.d).ok_or("Overflow racional")?,
        )
    }
    fn neg(self) -> Result<Self> {
        Self::new(self.n.checked_neg().ok_or("Overflow racional")?, self.d)
    }
    fn inverse(self) -> Result<Self> {
        Self::new(self.d, self.n)
    }
    fn text(self) -> String {
        if self.d == 1 {
            self.n.to_string()
        } else {
            format!("{}/{}", self.n, self.d)
        }
    }
}
type Monomial = Vec<(String, u16)>;
#[derive(Clone, Debug, PartialEq, Eq)]
struct Polynomial {
    terms: BTreeMap<Monomial, Rational>,
}
impl Polynomial {
    fn constant(x: Rational) -> Self {
        let mut terms = BTreeMap::new();
        if x.n != 0 {
            terms.insert(Vec::new(), x);
        }
        Self { terms }
    }
    fn variable(s: String) -> Self {
        Self {
            terms: BTreeMap::from([(vec![(s, 1)], Rational::integer(1))]),
        }
    }
    fn add(&self, b: &Self) -> Result<Self> {
        let mut result = self.clone();
        for (k, v) in &b.terms {
            let sum = result
                .terms
                .get(k)
                .copied()
                .unwrap_or(Rational::integer(0))
                .add(*v)?;
            if sum.n == 0 {
                result.terms.remove(k);
            } else {
                result.terms.insert(k.clone(), sum);
            }
        }
        if result.terms.len() > 4096 {
            return err("Polinômio excede 4096 termos");
        }
        Ok(result)
    }
    fn neg(&self) -> Result<Self> {
        Ok(Self {
            terms: self
                .terms
                .iter()
                .map(|(k, v)| Ok((k.clone(), v.neg()?)))
                .collect::<Result<_>>()?,
        })
    }
    fn mul(&self, b: &Self) -> Result<Self> {
        if self.terms.len().saturating_mul(b.terms.len()) > 65536 {
            return err("Produto polinomial excede orçamento");
        }
        let mut result = Self::constant(Rational::integer(0));
        for (a, x) in &self.terms {
            for (b, y) in &b.terms {
                let mut powers: BTreeMap<String, u16> = a.iter().cloned().collect();
                for (v, n) in b {
                    let entry = powers.entry(v.clone()).or_default();
                    *entry = entry.checked_add(*n).ok_or("Grau excedido")?;
                }
                if powers.values().map(|x| *x as usize).sum::<usize>() > 64 {
                    return err("Grau máximo 64");
                }
                let key: Vec<_> = powers.into_iter().collect();
                let prod = x.mul(*y)?;
                let sum = result
                    .terms
                    .get(&key)
                    .copied()
                    .unwrap_or(Rational::integer(0))
                    .add(prod)?;
                if sum.n == 0 {
                    result.terms.remove(&key);
                } else {
                    result.terms.insert(key, sum);
                }
                if result.terms.len() > 4096 {
                    return err("Polinômio excede 4096 termos");
                }
            }
        }
        Ok(result)
    }
    fn power(&self, n: usize) -> Result<Self> {
        if n > 32 {
            return err("Expoente máximo 32");
        }
        let mut out = Self::constant(Rational::integer(1));
        for _ in 0..n {
            out = out.mul(self)?;
        }
        Ok(out)
    }
    fn scalar(&self) -> Result<Rational> {
        if self.terms.is_empty() {
            return Ok(Rational::integer(0));
        }
        if self.terms.len() == 1 && self.terms.keys().next().unwrap().is_empty() {
            Ok(*self.terms.values().next().unwrap())
        } else {
            err("Divisor precisa ser constante racional")
        }
    }
    fn derivative(&self, var: &str) -> Result<Self> {
        let mut result = Self::constant(Rational::integer(0));
        for (m, c) in &self.terms {
            if let Some((idx, (_, power))) = m.iter().enumerate().find(|(_, v)| v.0 == var) {
                let mut key = m.clone();
                let factor = *power;
                if factor == 1 {
                    key.remove(idx);
                } else {
                    key[idx].1 -= 1;
                }
                let term = Self {
                    terms: BTreeMap::from([(key, c.mul(Rational::integer(factor as i128))?)]),
                };
                result = result.add(&term)?;
            }
        }
        Ok(result)
    }
    fn text(&self) -> String {
        if self.terms.is_empty() {
            return "0".into();
        }
        self.terms
            .iter()
            .map(|(m, c)| {
                let mut s = c.text();
                for (v, p) in m {
                    s.push('*');
                    s.push_str(v);
                    if *p > 1 {
                        s.push_str(&format!("**{p}"));
                    }
                }
                s
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }
    fn eval(&self, vars: &BTreeMap<String, f64>) -> Result<f64> {
        let mut out = 0.;
        for (m, c) in &self.terms {
            let mut v = c.n as f64 / c.d as f64;
            for (name, power) in m {
                v *= vars
                    .get(name)
                    .ok_or_else(|| format!("Variável ausente: {name}"))?
                    .powi(*power as i32);
            }
            out += v;
        }
        if out.is_finite() {
            Ok(out)
        } else {
            err("Resultado não finito")
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
enum Token {
    Num(i128),
    Name(String),
    Plus,
    Minus,
    Mul,
    Div,
    Pow,
    Left,
    Right,
    End,
}
struct PolyParser {
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
}
impl PolyParser {
    fn parse(text: &str) -> Result<Polynomial> {
        if text.len() > 4096 {
            return err("Expressão excede 4096 bytes");
        }
        let mut tokens = Vec::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                ' ' | '\t' | '\n' | '\r' => {}
                '+' => tokens.push(Token::Plus),
                '-' => tokens.push(Token::Minus),
                '*' => {
                    if chars.peek() == Some(&'*') {
                        chars.next();
                        tokens.push(Token::Pow);
                    } else {
                        tokens.push(Token::Mul);
                    }
                }
                '^' => tokens.push(Token::Pow),
                '/' => tokens.push(Token::Div),
                '(' => tokens.push(Token::Left),
                ')' => tokens.push(Token::Right),
                '0'..='9' => {
                    let mut s = c.to_string();
                    while chars.peek().is_some_and(char::is_ascii_digit) {
                        s.push(chars.next().unwrap());
                    }
                    tokens.push(Token::Num(s.parse().map_err(|_| "Inteiro excede i128")?));
                }
                c if c.is_alphabetic() || c == '_' => {
                    let mut s = c.to_string();
                    while chars
                        .peek()
                        .is_some_and(|c| c.is_alphanumeric() || *c == '_')
                    {
                        s.push(chars.next().unwrap());
                    }
                    tokens.push(Token::Name(s));
                }
                _ => return err("Apenas polinômios racionais são aceitos"),
            }
        }
        tokens.push(Token::End);
        let mut p = Self {
            tokens,
            pos: 0,
            depth: 0,
        };
        let result = p.expression()?;
        if p.tokens[p.pos] != Token::End {
            return err("Expressão polinomial incompleta");
        }
        Ok(result)
    }
    fn expression(&mut self) -> Result<Polynomial> {
        let mut x = self.product()?;
        loop {
            match self.tokens[self.pos] {
                Token::Plus => {
                    self.pos += 1;
                    x = x.add(&self.product()?)?;
                }
                Token::Minus => {
                    self.pos += 1;
                    x = x.add(&self.product()?.neg()?)?;
                }
                _ => break,
            }
        }
        Ok(x)
    }
    fn product(&mut self) -> Result<Polynomial> {
        let mut x = self.unary()?;
        loop {
            match self.tokens[self.pos] {
                Token::Mul => {
                    self.pos += 1;
                    x = x.mul(&self.unary()?)?;
                }
                Token::Div => {
                    self.pos += 1;
                    let y = self.unary()?.scalar()?.inverse()?;
                    x = x.mul(&Polynomial::constant(y))?;
                }
                _ => break,
            }
        }
        Ok(x)
    }
    fn unary(&mut self) -> Result<Polynomial> {
        self.depth += 1;
        if self.depth > 64 {
            return err("AST excede profundidade 64");
        }
        let r = match self.tokens[self.pos] {
            Token::Plus => {
                self.pos += 1;
                self.unary()
            }
            Token::Minus => {
                self.pos += 1;
                self.unary()?.neg()
            }
            _ => self.power(),
        };
        self.depth -= 1;
        r
    }
    fn power(&mut self) -> Result<Polynomial> {
        let mut x = self.atom()?;
        if self.tokens[self.pos] == Token::Pow {
            self.pos += 1;
            let n = self.unary()?.scalar()?;
            if n.d != 1 || n.n < 0 || n.n > 32 {
                return err("Expoente precisa ser inteiro 0..32");
            }
            x = x.power(n.n as usize)?;
        }
        Ok(x)
    }
    fn atom(&mut self) -> Result<Polynomial> {
        match self.tokens[self.pos].clone() {
            Token::Num(x) => {
                self.pos += 1;
                Ok(Polynomial::constant(Rational::integer(x)))
            }
            Token::Name(s) => {
                self.pos += 1;
                Ok(Polynomial::variable(s))
            }
            Token::Left => {
                self.pos += 1;
                let x = self.expression()?;
                if self.tokens[self.pos] != Token::Right {
                    return err("Parêntese não fechado");
                }
                self.pos += 1;
                Ok(x)
            }
            _ => err("Operando polinomial esperado"),
        }
    }
}
fn symbolic_equivalence_check(a: &str, b: &str) -> Result<Json> {
    let a = PolyParser::parse(a)?;
    let b = PolyParser::parse(b)?;
    let difference = a.add(&b.neg()?)?;
    Ok(obj([
        ("proved", Json::Bool(difference.terms.is_empty())),
        (
            "domain",
            js("Q[variáveis], i128 verificado; não IEEE-754 nem Python geral"),
        ),
        ("difference", js(difference.text())),
    ]))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b > 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}
fn powmod(mut a: u64, mut n: u64, m: u64) -> u64 {
    let mut r = 1 % m;
    while n > 0 {
        if n & 1 == 1 {
            r = mulmod(r, a, m);
        }
        a = mulmod(a, a, m);
        n >>= 1;
    }
    r
}
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'base: for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        if a % n == 0 {
            continue;
        }
        let mut x = powmod(a % n, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'base;
            }
        }
        return false;
    }
    true
}
fn factor(n: u64) -> Result<BTreeMap<u64, u32>> {
    if n == 0 {
        return err("Fatoração de zero indefinida");
    }
    fn split(n: u64, out: &mut BTreeMap<u64, u32>, budget: &mut usize) -> Result<()> {
        if n == 1 {
            return Ok(());
        }
        if is_prime(n) {
            *out.entry(n).or_default() += 1;
            return Ok(());
        }
        if n % 2 == 0 {
            *out.entry(2).or_default() += 1;
            return split(n / 2, out, budget);
        }
        for c in 1..64u64 {
            let mut x = 2;
            let mut y = 2;
            let mut d = 1;
            let step = |x: u64| ((mulmod(x, x, n) as u128 + c as u128) % n as u128) as u64;
            while d == 1 {
                if *budget == 0 {
                    return err("Fatoração excedeu orçamento de 2 milhões de passos");
                }
                *budget -= 1;
                x = step(x);
                y = step(step(y));
                d = gcd(x.abs_diff(y), n);
            }
            if d != n {
                split(d, out, budget)?;
                return split(n / d, out, budget);
            }
        }
        err("Pollard rho: nenhum divisor no orçamento")
    }
    let mut out = BTreeMap::new();
    split(n, &mut out, &mut 2_000_000)?;
    Ok(out)
}
fn uint(j: &Json) -> Result<u64> {
    match j {
        Json::Str(s) => s.parse().map_err(|_| "Inteiro u64 inválido".into()),
        Json::Num(n) if *n >= 0. && *n <= 9_007_199_254_740_991. && n.fract() == 0. => {
            Ok(*n as u64)
        }
        _ => err("Use string decimal para inteiros acima de 2^53-1"),
    }
}
fn exact_int(n: u128) -> Json {
    if n <= 9_007_199_254_740_991 {
        jn(n as f64)
    } else {
        js(n.to_string())
    }
}
fn numeric_vec(j: &Json) -> Result<Vec<f64>> {
    j.array()?.iter().map(Json::number).collect()
}
fn matrix(j: &Json) -> Result<Vec<Vec<f64>>> {
    let a = j
        .array()?
        .iter()
        .map(numeric_vec)
        .collect::<Result<Vec<_>>>()?;
    if a.is_empty()
        || a.len() > 256
        || a[0].is_empty()
        || a[0].len() > 256
        || a.iter().any(|r| r.len() != a[0].len())
    {
        return err("Matriz retangular 1..256 esperada");
    }
    Ok(a)
}
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = a.len();
    if a.iter().any(|r| r.len() != n) || b.len() != n {
        return err("Sistema quadrado esperado");
    }
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        if a[pivot][k].abs() < EPS {
            return err("Sistema singular ou mal condicionado");
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        let s = a[k][k];
        for j in k..n {
            a[k][j] /= s;
        }
        b[k] /= s;
        for i in 0..n {
            if i == k {
                continue;
            }
            let v = a[i][k];
            for j in k..n {
                a[i][j] -= v * a[k][j];
            }
            b[i] -= v * b[k];
        }
    }
    if b.iter().any(|x| !x.is_finite()) {
        return err("Resultado não finito");
    }
    Ok(b)
}
fn determinant(mut a: Vec<Vec<f64>>) -> Result<f64> {
    let n = a.len();
    if a.iter().any(|r| r.len() != n) {
        return err("Matriz quadrada esperada");
    }
    let mut d = 1.;
    for k in 0..n {
        let i = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        if a[i][k].abs() < EPS {
            return Ok(0.);
        }
        if i != k {
            a.swap(i, k);
            d = -d;
        }
        d *= a[k][k];
        for i in k + 1..n {
            let r = a[i][k] / a[k][k];
            for j in k + 1..n {
                a[i][j] -= r * a[k][j];
            }
        }
    }
    if !d.is_finite() {
        return err("Determinante não finito");
    }
    Ok(d)
}
fn sat_solve(
    clauses: &[Vec<i32>],
    assignment: &mut BTreeMap<i32, bool>,
    budget: &mut usize,
) -> Result<bool> {
    if *budget == 0 {
        return err("SAT excedeu orçamento");
    }
    *budget -= 1;
    let mut chosen = None;
    for c in clauses {
        let mut satisfied = false;
        let mut unassigned = None;
        for &lit in c {
            match assignment.get(&lit.abs()) {
                Some(v) if *v == (lit > 0) => {
                    satisfied = true;
                    break;
                }
                None => unassigned = Some(lit.abs()),
                _ => {}
            }
        }
        if !satisfied {
            if unassigned.is_none() {
                return Ok(false);
            }
            chosen = unassigned;
        }
    }
    let Some(var) = chosen else { return Ok(true) };
    for value in [true, false] {
        assignment.insert(var, value);
        if sat_solve(clauses, assignment, budget)? {
            return Ok(true);
        }
        assignment.remove(&var);
    }
    Ok(false)
}
const MATH_OPS: &[&str] = &[
    "system.arithmetic",
    "system.equivalence",
    "nt.is_prime",
    "nt.factor",
    "nt.gcd",
    "nt.lcm",
    "nt.phi",
    "nt.mobius",
    "nt.sigma",
    "nt.sieve",
    "nt.fib",
    "nt.modpow",
    "nt.modinv",
    "nt.crt",
    "comb.binom",
    "comb.catalan",
    "comb.stirling2",
    "comb.bell",
    "comb.derangements",
    "la.solve",
    "la.det",
    "la.inverse",
    "la.matmul",
    "la.eigen_power",
    "st.mean",
    "st.variance",
    "st.correlation",
    "it.entropy",
    "lg.dpll",
    "gr.bellman_ford",
    "gr.floyd_warshall",
    "de.heat1d",
    "de.wave1d",
    "ds.lorenz",
    "nat.derivada",
    "vsa.bind",
];
fn calcular(operation: &str, args: &[Json]) -> Result<Json> {
    let need = |n| {
        if args.len() == n {
            Ok(())
        } else {
            err(format!("{operation}: esperados {n} argumentos"))
        }
    };
    let at = |i: usize| args.get(i).ok_or_else(|| "Argumento ausente".to_string());
    match operation{
        "system.arithmetic"=>{need(1)?;let p=PolyParser::parse(at(0)?.string()?)?;Ok(js(p.scalar()?.text()))},
        "system.equivalence"=>{need(2)?;symbolic_equivalence_check(at(0)?.string()?,at(1)?.string()?)},
        "nt.is_prime"=>{need(1)?;Ok(Json::Bool(is_prime(uint(at(0)?)?)))},
        "nt.factor"=>{need(1)?;Ok(Json::Obj(factor(uint(at(0)?)?)?.into_iter().map(|(p,e)|(p.to_string(),jn(e as f64))).collect()))},
        "nt.gcd"|"nt.lcm"=>{need(2)?;let a=uint(at(0)?)?;let b=uint(at(1)?)?;Ok(exact_int(if operation=="nt.gcd"{gcd(a,b)as u128}else if a==0||b==0{0}else{(a/gcd(a,b))as u128*b as u128}))},
        "nt.phi"|"nt.mobius"|"nt.sigma"=>{need(1)?;let n=uint(at(0)?)?;let f=factor(n)?;if operation=="nt.mobius"{return Ok(jn(if f.values().any(|e|*e>1){0.}else if f.len()%2==0{1.}else{-1.}));}let mut result=if operation=="nt.phi"{n as u128}else{1};for(p,e)in f{if operation=="nt.phi"{result=result/p as u128*(p-1)as u128;}else{let sum=(0..=e).try_fold(0u128,|s,k|(p as u128).checked_pow(k).and_then(|x|s.checked_add(x))).ok_or("Overflow sigma")?;result=result.checked_mul(sum).ok_or("Overflow sigma")?;}}Ok(exact_int(result))},
        "nt.sieve"=>{need(1)?;let n=uint(at(0)?)?as usize;if n>2_000_000{return err("Crivo máximo 2 milhões");}let mut prime=vec![true;n+1];prime[0]=false;if n>0{prime[1]=false;}for i in 2..=n{if prime[i]&&i<=n/i{for j in(i*i..=n).step_by(i){prime[j]=false;}}}Ok(Json::Arr(prime.iter().enumerate().filter(|(_,v)|**v).map(|(i,_)|jn(i as f64)).collect()))},
        "nt.fib"=>{need(1)?;let n=uint(at(0)?)?;if n>186{return err("Fibonacci excede u128");}let(mut a,mut b)=(0u128,1u128);for i in 0..n{if i==n-1{return Ok(exact_int(b));}let next=a.checked_add(b).ok_or("Overflow Fibonacci")?;a=b;b=next;}Ok(exact_int(a))},
        "nt.modpow"=>{need(3)?;let m=uint(at(2)?)?;if m==0{return err("Módulo zero");}Ok(exact_int(powmod(uint(at(0)?)?,uint(at(1)?)?,m)as u128))},
        "nt.modinv"=>{need(2)?;let a=uint(at(0)?)?as i128;let m=uint(at(1)?)?as i128;if m<2{return err("Módulo >=2 esperado");}let(mut oldr,mut r,mut olds,mut s)=(a,m,1i128,0i128);while r!=0{let q=oldr/r;(oldr,r)=(r,oldr-q*r);(olds,s)=(s,olds-q*s);}if oldr!=1{return err("Inverso inexistente");}Ok(exact_int(olds.rem_euclid(m)as u128))},
        "nt.crt"=>{need(2)?;let rs=at(0)?.array()?;let ms=at(1)?.array()?;if rs.is_empty()||rs.len()!=ms.len()||rs.len()>32{return err("CRT: listas de mesmo tamanho 1..32");}let(mut x,mut modulus)=(0u128,1u128);for(r,m)in rs.iter().zip(ms){let m=uint(m)?;let r=uint(r)?;if m<2||modulus>u64::MAX as u128{return err("CRT: módulo excedido");}let inv=uint(&calcular("nt.modinv",&[exact_int(modulus%m as u128),exact_int(m as u128)])?)?;let residue=((r as u128+m as u128-(x%m as u128))%m as u128)as u64;let k=mulmod(residue,inv,m)as u128;x=x.checked_add(modulus.checked_mul(k).ok_or("CRT overflow")?).ok_or("CRT overflow")?;modulus=modulus.checked_mul(m as u128).ok_or("CRT overflow")?;x%=modulus;}Ok(Json::Arr(vec![exact_int(x),exact_int(modulus)]))},
        "comb.binom"|"comb.catalan"=>{need(if operation=="comb.binom"{2}else{1})?;let n=uint(at(0)?)?;let(n,k)=if operation=="comb.catalan"{if n>64{return err("Catalan máximo 64");}(n*2,n)}else{(n,uint(at(1)?)?)};if n>128{return err("Binomial máximo n=128");}if k>n{return Ok(jn(0.));}let mut row=vec![0u128;k as usize+1];row[0]=1;for i in 1..=n{for j in(1..=i.min(k)).rev(){row[j as usize]=row[j as usize].checked_add(row[j as usize-1]).ok_or("Binomial overflow")?;}}let mut r=row[k as usize];if operation=="comb.catalan"{r/=k as u128+1;}Ok(exact_int(r))},
        "comb.stirling2"|"comb.bell"=>{need(if operation=="comb.bell"{1}else{2})?;let n=uint(at(0)?)?as usize;if n>32{return err("Stirling/Bell: n máximo 32");}let k=if operation=="comb.bell"{n}else{uint(at(1)?)?as usize};if k>n{return Ok(jn(0.));}let mut row=vec![0u128;n+1];row[0]=1;for i in 1..=n{for j in(1..=i).rev(){row[j]=row[j].checked_mul(j as u128).and_then(|v|v.checked_add(row[j-1])).ok_or("Stirling overflow")?;}row[0]=0;}let r=if operation=="comb.bell"{row.iter().sum()}else{row[k]};Ok(exact_int(r))},
        "comb.derangements"=>{need(1)?;let n=uint(at(0)?)?;if n>34{return err("Desarranjos: n máximo 34");}let(mut a,mut b)=(1u128,0u128);for i in 2..=n{let c=(i as u128-1).checked_mul(a.checked_add(b).ok_or("Overflow")?).ok_or("Overflow")?;a=b;b=c;}Ok(exact_int(if n==0{1}else{b}))},
        "la.solve"=>{need(2)?;Ok(ja(&solve(matrix(at(0)?)?,numeric_vec(at(1)?)?)?))},
        "la.det"=>{need(1)?;Ok(jn(determinant(matrix(at(0)?)?)?))},
        "la.inverse"=>{need(1)?;let a=matrix(at(0)?)?;let n=a.len();let mut inverse=vec![vec![0.;n];n];for j in 0..n{let mut unit=vec![0.;n];unit[j]=1.;let col=solve(a.clone(),unit)?;for i in 0..n{inverse[i][j]=col[i];}}Ok(Json::Arr(inverse.iter().map(|r|ja(r)).collect()))},
        "la.matmul"=>{need(2)?;let a=matrix(at(0)?)?;let b=matrix(at(1)?)?;if a[0].len()!=b.len(){return err("Matrizes incompatíveis");}let mut out=vec![vec![0.;b[0].len()];a.len()];for i in 0..a.len(){for k in 0..b.len(){for j in 0..b[0].len(){out[i][j]+=a[i][k]*b[k][j];}}}if out.iter().flatten().any(|x|!x.is_finite()){return err("Produto não finito");}Ok(Json::Arr(out.iter().map(|r|ja(r)).collect()))},
        "la.eigen_power"=>{need(1)?;let a=matrix(at(0)?)?;let n=a.len();if a[0].len()!=n{return err("Matriz quadrada esperada");}let mut v=vec![1./(n as f64).sqrt();n];for _ in 0..256{let y:Vec<_>=a.iter().map(|r|dot(r,&v)).collect();let s=norm(&y);if s<EPS{return err("Iteração de potência degenerada");}v=y.iter().map(|x|x/s).collect();}let av:Vec<_>=a.iter().map(|r|dot(r,&v)).collect();let eigen=dot(&v,&av);let residual=norm(&av.iter().zip(&v).map(|(a,b)|a-eigen*b).collect::<Vec<_>>());Ok(obj([("eigenvalue",jn(eigen)),("vector",ja(&v)),("residual",jn(residual)),("exact",Json::Bool(false))]))},
        "st.mean"|"st.variance"|"it.entropy"=>{need(1)?;let v=numeric_vec(at(0)?)?;if v.is_empty(){return err("Lista vazia");}let mean=v.iter().sum::<f64>()/v.len()as f64;let value=match operation{"st.mean"=>mean,"st.variance"=>v.iter().map(|x|(x-mean).powi(2)).sum::<f64>()/v.len()as f64,_=>{if v.iter().any(|x|*x<0.)||(v.iter().sum::<f64>()-1.).abs()>1e-8{return err("Probabilidades precisam somar 1");}-v.iter().filter(|x|**x>0.).map(|x|x*x.log2()).sum::<f64>()}};if !value.is_finite(){return err("Estatística não finita");}Ok(jn(value))},
        "st.correlation"=>{need(2)?;let a=numeric_vec(at(0)?)?;let b=numeric_vec(at(1)?)?;if a.len()!=b.len()||a.is_empty(){return err("Vetores incompatíveis");}let ma=a.iter().sum::<f64>()/a.len()as f64;let mb=b.iter().sum::<f64>()/b.len()as f64;let a:Vec<_>=a.iter().map(|x|x-ma).collect();let b:Vec<_>=b.iter().map(|x|x-mb).collect();let denominator=norm(&a)*norm(&b);if denominator<EPS{return err("Correlação indefinida");}Ok(jn(dot(&a,&b)/denominator))},
        "lg.dpll"=>{need(1)?;let clauses=at(0)?.array()?.iter().map(|c|c.array()?.iter().map(|x|{let v=x.number()?;if v.fract()!=0.||v==0.||v.abs()>128.{return err("Literal SAT deve ser ±1..128");}Ok(v as i32)}).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;if clauses.len()>1024{return err("Máximo 1024 cláusulas");}let mut assignment=BTreeMap::new();let yes=sat_solve(&clauses,&mut assignment,&mut 100_000)?;Ok(obj([("satisfiable",Json::Bool(yes)),("assignment",if yes{Json::Obj(assignment.into_iter().map(|(k,v)|(k.to_string(),Json::Bool(v))).collect())}else{Json::Null})]))},
        "gr.bellman_ford"=>{need(3)?;let n=at(0)?.usize()?;let edges=at(1)?.array()?;let source=at(2)?.usize()?;if n==0||n>256||source>=n||edges.len()>65536{return err("Grafo excede limites");}let edges=edges.iter().map(|e|{let e=e.array()?;if e.len()!=3{return err("Aresta deve ser [origem,destino,peso]");}let(a,b,w)=(e[0].usize()?,e[1].usize()?,e[2].number()?);if a>=n||b>=n{return err("Vértice inválido");}Ok((a,b,w))}).collect::<Result<Vec<_>>>()?;let mut distance=vec![f64::INFINITY;n];distance[source]=0.;for i in 0..n{let mut changed=false;for &(a,b,w)in &edges{if distance[a]+w<distance[b]{distance[b]=distance[a]+w;changed=true;if i==n-1{return err("Ciclo negativo alcançável");}}}if !changed{break;}}Ok(Json::Arr(distance.iter().map(|x|if x.is_finite(){jn(*x)}else{Json::Null}).collect()))},
        "gr.floyd_warshall"=>{need(1)?;let rows=at(0)?.array()?;let n=rows.len();if n==0||n>128{return err("Grafo máximo 128");}let mut d=Vec::new();for row in rows{let r=row.array()?;if r.len()!=n{return err("Matriz quadrada esperada");}d.push(r.iter().map(|v|if v==&Json::Null{Ok(f64::INFINITY)}else{v.number()}).collect::<Result<Vec<_>>>()?);}for i in 0..n{d[i][i]=d[i][i].min(0.);}for k in 0..n{for i in 0..n{for j in 0..n{d[i][j]=d[i][j].min(d[i][k]+d[k][j]);}}}if(0..n).any(|i|d[i][i]<0.){return err("Ciclo negativo");}Ok(Json::Arr(d.iter().map(|r|Json::Arr(r.iter().map(|x|if x.is_finite(){jn(*x)}else{Json::Null}).collect())).collect()))},
        "de.heat1d"|"de.wave1d"=>{need(3)?;let mut u=numeric_vec(at(0)?)?;let steps=at(1)?.usize()?;let ratio=at(2)?.number()?;let max=if operation=="de.heat1d"{0.5}else{1.};if u.len()<3||u.len()>4096||steps>10000||!(0. ..=max).contains(&ratio){return err("Malha/tempo inválido ou condição CFL violada");}let mut previous=u.clone();for _ in 0..steps{let mut next=u.clone();for j in 1..u.len()-1{let lap=u[j-1]-2.*u[j]+u[j+1];next[j]=if operation=="de.heat1d"{u[j]+ratio*lap}else{2.*u[j]-previous[j]+ratio*ratio*lap};}previous=u;u=next;if u.iter().any(|x|!x.is_finite()){return err("Simulação não finita");}}Ok(ja(&u))},
        "ds.lorenz"=>{need(3)?;let mut state=numeric_vec(at(0)?)?;let steps=at(1)?.usize()?;let dt=at(2)?.number()?;if state.len()!=3||steps>100000||dt<=0.||dt>0.02{return err("Lorenz: estado 3D, passos <=100000, dt em (0,.02]");}let f=|v:&[f64]|vec![10.*(v[1]-v[0]),v[0]*(28.-v[2])-v[1],v[0]*v[1]-8./3.*v[2]];for _ in 0..steps{let k1=f(&state);let temp:Vec<_>=state.iter().zip(&k1).map(|(s,k)|s+dt*k/2.).collect();let k2=f(&temp);let temp:Vec<_>=state.iter().zip(&k2).map(|(s,k)|s+dt*k/2.).collect();let k3=f(&temp);let temp:Vec<_>=state.iter().zip(&k3).map(|(s,k)|s+dt*k).collect();let k4=f(&temp);for j in 0..3{state[j]+=dt*(k1[j]+2.*k2[j]+2.*k3[j]+k4[j])/6.;}if state.iter().any(|x|!x.is_finite()){return err("Lorenz não finito");}}Ok(ja(&state))},
        "vsa.bind"=>{need(2)?;Ok(ja(&vsa_bind(&numeric_vec(at(0)?)?,&numeric_vec(at(1)?)?)?))},
        "nat.derivada"=>{need(2)?;Ok(js(PolyParser::parse(at(0)?.string()?)?.derivative(at(1)?.string()?)?.text()))},
        _=>err(format!("Operação não portada: {operation}. Use --coverage; nenhum fallback Python é executado."))
    }
}

// Importação incremental e limitada. Uma extensão binária nunca vira texto por
// engano. Não há desserialização de pickle, execução de código ou rede implícita.
fn pair_from_record(j: &Json) -> Result<Vec<Pair>> {
    let mut pairs = Vec::new();
    if let Json::Arr(a) = j {
        for x in a {
            pairs.extend(pair_from_record(x)?);
        }
        return Ok(pairs);
    }
    if let Json::Str(s) = j {
        return Ok(vec![Pair {
            input: format!("Continue: {}", s.chars().take(64).collect::<String>()),
            output: s.clone(),
            trusted: false,
        }]);
    }
    if let Json::Obj(o) = j {
        for field in ["messages", "conversations"] {
            if let Some(Json::Arr(messages)) = o.get(field) {
                let mut pending = None;
                for message in messages {
                    let role = message
                        .get("role")
                        .string()
                        .or_else(|_| message.get("from").string())
                        .unwrap_or("");
                    let content = message
                        .get("content")
                        .string()
                        .or_else(|_| message.get("value").string())
                        .unwrap_or("");
                    if role == "user" || role == "human" {
                        pending = Some(content.to_string());
                    } else if role == "assistant" || role == "gpt" {
                        if let Some(input) = pending.take() {
                            pairs.push(Pair {
                                input,
                                output: content.into(),
                                trusted: false,
                            });
                        }
                    }
                }
                return validate_pairs(pairs);
            }
        }
        let input = ["instruction", "question", "prompt", "input"]
            .iter()
            .find_map(|k| o.get(*k).and_then(|x| x.string().ok()));
        let output = ["output", "answer", "response", "completion", "target"]
            .iter()
            .find_map(|k| o.get(*k).and_then(|x| x.string().ok()));
        if let (Some(input), Some(output)) = (input, output) {
            let mut input = input.to_string();
            if o.contains_key("instruction") {
                if let Some(Json::Str(extra)) = o.get("input") {
                    if !extra.is_empty() {
                        input.push('\n');
                        input.push_str(extra);
                    }
                }
            }
            pairs.push(Pair {
                input,
                output: output.into(),
                trusted: false,
            });
        } else if let Some(Json::Str(text)) = o.get("text") {
            pairs = pair_from_record(&js(text))?;
        } else if let Some(data) = o.get("data").or_else(|| o.get("train")) {
            pairs = pair_from_record(data)?;
        } else {
            return err("Registro sem par instruction/output, question/answer, prompt/completion, mensagens ou text");
        }
    } else {
        return err("Registro de dataset inválido");
    }
    validate_pairs(pairs)
}
fn validate_pairs(pairs: Vec<Pair>) -> Result<Vec<Pair>> {
    for p in &pairs {
        if p.input.len() > 16384
            || p.output.len() > 16384
            || p.input.trim().is_empty()
            || p.output.trim().is_empty()
        {
            return err("Dataset: campos vazios ou maiores que 16 KiB");
        }
    }
    Ok(pairs)
}
fn csv_rows(text: &str, delimiter: char) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    let mut closed = false;
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                field.push(c);
            }
        } else {
            match c {
                '"' if field.is_empty() && !closed => quoted = true,
                c if c == delimiter => {
                    row.push(std::mem::take(&mut field));
                    closed = false;
                }
                '\n' => {
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                    closed = false;
                }
                '\r' if chars.peek() == Some(&'\n') => {}
                _ if closed => return err("CSV: caracteres depois de aspas"),
                _ => field.push(c),
            }
        }
        if field.len() > 16384 {
            return err("CSV: campo excede 16 KiB");
        }
    }
    if quoted {
        return err("CSV: aspas não fechadas");
    }
    if !row.is_empty() || !field.is_empty() || closed {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}
fn import_dataset(path: &Path, limit: usize) -> Result<Vec<Pair>> {
    if limit == 0 || limit > 10000 {
        return err("Limite deve ser 1..10000 registros");
    }
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if !["json", "jsonl", "ndjson", "csv", "tsv", "txt", "md"].contains(&ext.as_str()) {
        return err(format!("Formato .{ext} ainda não portado. Rust aceita JSON, JSONL, NDJSON, CSV, TSV, TXT e MD; --coverage detalha as diferenças."));
    }
    let data = read_bounded(path)?;
    let text = std::str::from_utf8(&data)
        .map_err(|_| "Dataset requer UTF-8")?
        .trim_start_matches('\u{feff}');
    let mut reservoir = Reservoir::new(limit, 41);
    let mut accept = |pairs: Vec<Pair>| {
        for p in pairs {
            reservoir.push(p);
        }
    };
    match ext.as_str() {
        "json" => accept(pair_from_record(&Json::parse(text)?)?),
        "jsonl" | "ndjson" => {
            for (i, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let pairs = Json::parse(line)
                    .and_then(|j| pair_from_record(&j))
                    .map_err(|e| format!("Linha {}: {e}", i + 1))?;
                accept(pairs);
            }
        }
        "csv" | "tsv" => {
            let rows = csv_rows(text, if ext == "csv" { ',' } else { '\t' })?;
            let header = rows.first().ok_or("CSV vazio")?;
            if header.iter().collect::<BTreeSet<_>>().len() != header.len() {
                return err("CSV com cabeçalho duplicado");
            }
            for (i, row) in rows.iter().enumerate().skip(1) {
                if row.len() == 1 && row[0].is_empty() {
                    continue;
                }
                if row.len() != header.len() {
                    return err(format!("CSV linha {}: número de colunas divergente", i + 1));
                }
                let j = Json::Obj(
                    header
                        .iter()
                        .cloned()
                        .zip(row.iter().cloned().map(js))
                        .collect(),
                );
                accept(pair_from_record(&j)?);
            }
        }
        _ => {
            for paragraph in text.split("\n\n") {
                let paragraph = paragraph.trim();
                if paragraph.is_empty() {
                    continue;
                }
                let chars: Vec<_> = paragraph.chars().collect();
                for chunk in chars.chunks(2048) {
                    let output: String = chunk.iter().collect();
                    let input = format!("Continue: {}", chunk.iter().take(64).collect::<String>());
                    accept(vec![Pair {
                        input,
                        output,
                        trusted: false,
                    }]);
                }
            }
        }
    }
    if reservoir.items.is_empty() {
        return err("Nenhum exemplo importável");
    }
    validate_pairs(reservoir.items)
}
fn split_dataset(pairs: &[Pair]) -> Result<(Vec<Pair>, Vec<Pair>)> {
    // Agrupa prompts idênticos antes do holdout para evitar vazamento trivial.
    let mut groups: BTreeMap<String, Vec<Pair>> = BTreeMap::new();
    for p in pairs {
        groups
            .entry(p.input.trim().to_string())
            .or_default()
            .push(p.clone());
    }
    if groups.len() < 2 {
        return err("Holdout exige ao menos dois prompts distintos");
    }
    let mut ordered: Vec<_> = groups.into_iter().collect();
    ordered.sort_by_key(|(s, _)| digest(s.as_bytes()));
    let n = (ordered.len() / 5).max(1);
    let mut train = Vec::new();
    let mut holdout = Vec::new();
    for (i, (_, group)) in ordered.into_iter().enumerate() {
        if i < n {
            holdout.extend(group);
        } else {
            train.extend(group);
        }
    }
    Ok((train, holdout))
}
fn mean_loss_deadline(model: &NativeSeq2Seq, pairs: &[Pair], deadline: Instant) -> Result<f64> {
    if pairs.is_empty() {
        return err("Avaliação vazia");
    }
    let mut loss = 0.;
    for p in pairs {
        if Instant::now() > deadline {
            return err("Prazo cooperativo de avaliação excedido; candidato descartado");
        }
        loss += model.loss_grad(&p.input, &zero_context(), &p.output)?.0;
    }
    Ok(loss / pairs.len() as f64)
}
#[derive(Clone, Debug)]
struct EvolutionReport {
    promoted: bool,
    before: f64,
    after: f64,
    train_count: usize,
    holdout_count: usize,
    steps: usize,
    dimension_before: usize,
    dimension_after: usize,
}
impl EvolutionReport {
    fn json(&self) -> Json {
        obj([
            ("promoted", Json::Bool(self.promoted)),
            ("holdout_before", jn(self.before)),
            ("holdout_after", jn(self.after)),
            ("train_records", jn(self.train_count as f64)),
            ("holdout_records", jn(self.holdout_count as f64)),
            ("updates", jn(self.steps as f64)),
            ("dimension_before", jn(self.dimension_before as f64)),
            ("dimension_after", jn(self.dimension_after as f64)),
        ])
    }
}
fn evolve_dataset(
    model: &mut NativeSeq2Seq,
    pairs: &[Pair],
    replay: &[Pair],
    restricted: bool,
    epochs: usize,
    resize: Option<usize>,
    seconds: f64,
) -> Result<EvolutionReport> {
    if epochs == 0 || epochs > 10 || !seconds.is_finite() || !(0.1..=60.).contains(&seconds) {
        return err("Evolução: épocas 1..10, prazo .1..60 s");
    }
    let deadline = Instant::now() + Duration::from_secs_f64(seconds);
    let (train, holdout) = split_dataset(pairs)?;
    let before = mean_loss_deadline(model, &holdout, deadline)?;
    let retention_before = if replay.is_empty() {
        0.
    } else {
        mean_loss_deadline(model, replay, deadline)?
    };
    let mut candidate = if let Some(h) = resize {
        model.resized(h)?
    } else {
        model.clone()
    };
    let mut steps = 0;
    let mut local_replay = replay.to_vec();
    'training: for _ in 0..epochs {
        for p in &train {
            if Instant::now() > deadline {
                break 'training;
            }
            let report = train_pair(
                &mut candidate,
                p,
                &local_replay,
                restricted,
                if restricted { 0.06 } else { 0.12 },
            )?;
            if report.accepted {
                steps += 1;
            }
            local_replay.push(p.clone());
            if local_replay.len() > 8 {
                local_replay.remove(0);
            }
        }
    }
    let after = mean_loss_deadline(&candidate, &holdout, deadline)?;
    let retention_after = if replay.is_empty() {
        0.
    } else {
        mean_loss_deadline(&candidate, replay, deadline)?
    };
    if Instant::now() > deadline {
        return err("Prazo cooperativo excedido; candidato descartado");
    }
    let promoted = steps > 0 && after + 1e-8 < before && retention_after <= retention_before + 0.02;
    let dimension_before = model.dimension;
    let dimension_after = if promoted {
        candidate.dimension
    } else {
        dimension_before
    };
    if promoted {
        *model = candidate;
    }
    Ok(EvolutionReport {
        promoted,
        before,
        after,
        train_count: train.len(),
        holdout_count: holdout.len(),
        steps,
        dimension_before,
        dimension_after,
    })
}

#[derive(Clone, Debug)]
struct BytePairEncoding {
    merges: Vec<(usize, usize)>,
}
impl BytePairEncoding {
    fn new() -> Self {
        Self { merges: Vec::new() }
    }
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens: Vec<_> = text.as_bytes().iter().map(|x| *x as usize).collect();
        for (i, &pair) in self.merges.iter().enumerate() {
            let mut next = Vec::new();
            let mut j = 0;
            while j < tokens.len() {
                if j + 1 < tokens.len() && (tokens[j], tokens[j + 1]) == pair {
                    next.push(256 + i);
                    j += 2;
                } else {
                    next.push(tokens[j]);
                    j += 1;
                }
            }
            tokens = next;
        }
        tokens
    }
    fn train(&mut self, text: &str, count: usize) -> Result<()> {
        if text.len() > 1024 * 1024 || count > 128 || self.merges.len() + count > 128 {
            return err("BPE: corpus <=1 MiB, até 128 merges totais");
        }
        let mut tokens = self.encode(text);
        for _ in 0..count {
            let mut freq = BTreeMap::new();
            for p in tokens.windows(2) {
                *freq.entry((p[0], p[1])).or_insert(0usize) += 1;
            }
            let Some((pair, n)) = freq
                .into_iter()
                .max_by_key(|(p, n)| (*n, std::cmp::Reverse(*p)))
            else {
                break;
            };
            if n < 2 {
                break;
            }
            let id = 256 + self.merges.len();
            self.merges.push(pair);
            let mut next = Vec::new();
            let mut j = 0;
            while j < tokens.len() {
                if j + 1 < tokens.len() && (tokens[j], tokens[j + 1]) == pair {
                    next.push(id);
                    j += 2;
                } else {
                    next.push(tokens[j]);
                    j += 1;
                }
            }
            tokens = next;
        }
        Ok(())
    }
    fn decode(&self, tokens: &[usize]) -> Result<String> {
        fn expand(id: usize, merges: &[(usize, usize)], out: &mut Vec<u8>) -> Result<()> {
            if out.len() > MAX_FILE {
                return err("BPE: expansão excede limite");
            }
            if id < 256 {
                out.push(id as u8);
            } else {
                let &(a, b) = merges.get(id - 256).ok_or("BPE: token inválido")?;
                if a >= id || b >= id {
                    return err("BPE: árvore cíclica");
                }
                expand(a, merges, out)?;
                expand(b, merges, out)?;
            }
            Ok(())
        }
        let mut out = Vec::new();
        for &t in tokens {
            expand(t, &self.merges, &mut out)?;
        }
        String::from_utf8(out).map_err(|_| "BPE: UTF-8 inválido".into())
    }
}
fn semantic_vector(text: &str, dimension: usize) -> Vec<f64> {
    let mut v = vec![0.; dimension];
    let lower = text.to_lowercase();
    for token in lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .take(1024)
    {
        let hash = sha256(token.as_bytes());
        let mut rng = Rng::new(u64::from_le_bytes(hash[..8].try_into().unwrap()));
        for x in &mut v {
            *x += if rng.next() & 1 == 0 { 1. } else { -1. };
        }
    }
    let n = norm(&v);
    if n > EPS {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}
fn vsa_bind(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    if a.len() != b.len() {
        return err("VSA: dimensões distintas");
    }
    Ok(a.iter().zip(b).map(|(a, b)| a * b).collect())
}

// O lock é do sistema operacional: encerrou o processo, o lock se libera.
// O arquivo permanece; não removê-lo evita a corrida entre inodes antigos/novos.
struct Lease {
    _file: File,
}
impl Lease {
    fn acquire(directory: &Path) -> Result<Self> {
        let path = directory.join(".rust.lock");
        if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
            return err("Lock não pode ser um link simbólico");
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(0);
        }
        let mut file = options.open(&path).map_err(ioerr)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            unsafe extern "C" {
                fn flock(fd: i32, operation: i32) -> i32;
            }
            // SAFETY: descriptor válido e vivo; flock não recebe ponteiros.
            if unsafe { flock(file.as_raw_fd(), 2 | 4) } != 0 {
                return err(format!("Perfil já aberto: {}", directory.display()));
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            return err("Locks de perfil não implementados nesta plataforma");
        }
        file.set_len(0).map_err(ioerr)?;
        writeln!(file, "{} {}", std::process::id(), now()).map_err(ioerr)?;
        file.sync_all().map_err(ioerr)?;
        Ok(Self { _file: file })
    }
}
struct EventHistory {
    path: PathBuf,
    head: String,
    sequence: u64,
}
impl EventHistory {
    fn open(path: PathBuf) -> Result<Self> {
        let mut h = Self {
            path,
            head: "0".repeat(64),
            sequence: 0,
        };
        if !h.path.exists() {
            return Ok(h);
        }
        let f = File::open(&h.path).map_err(ioerr)?;
        let mut reader = BufReader::new(f);
        loop {
            let mut line = String::new();
            let n = reader
                .by_ref()
                .take(256 * 1024)
                .read_line(&mut line)
                .map_err(ioerr)?;
            if n == 0 {
                break;
            }
            if !line.ends_with('\n') {
                return err("Histórico truncado ou evento maior que 256 KiB");
            }
            let event = Json::parse(line.trim_end())?;
            let body = event.get("body");
            let sequence = body
                .get("sequence")
                .string()?
                .parse::<u64>()
                .map_err(|_| "Sequência inválida")?;
            if sequence != h.sequence + 1
                || body.get("previous").string()? != h.head
                || event.get("hash").string()? != digest(body.encode().as_bytes())
            {
                return err(format!("Histórico adulterado no evento {sequence}"));
            }
            h.sequence = sequence;
            h.head = event.get("hash").string()?.into();
        }
        Ok(h)
    }
    fn contains_anchor(&self, hash: &str) -> Result<bool> {
        if hash == "0".repeat(64) {
            return Ok(true);
        }
        let mut reader = BufReader::new(File::open(&self.path).map_err(ioerr)?);
        loop {
            let mut line = String::new();
            let count = reader
                .by_ref()
                .take(256 * 1024)
                .read_line(&mut line)
                .map_err(ioerr)?;
            if count == 0 {
                return Ok(false);
            }
            if !line.ends_with('\n') {
                return err("Histórico truncado");
            }
            let event = Json::parse(line.trim_end())?;
            if event.get("hash").string()? == hash {
                return Ok(true);
            }
        }
    }
    fn append(&mut self, kind: &str, payload: Json) -> Result<()> {
        let body = obj([
            ("sequence", js((self.sequence + 1).to_string())),
            ("previous", js(&self.head)),
            ("utc_epoch", js(now().to_string())),
            ("kind", js(kind)),
            ("payload", payload),
        ]);
        let hash = digest(body.encode().as_bytes());
        let event = obj([("body", body), ("hash", js(&hash))]);
        let line = event.encode() + "\n";
        if line.len() > 256 * 1024 {
            return err("Evento excede 256 KiB");
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(ioerr)?;
        if f.metadata().map_err(ioerr)?.len() + line.len() as u64 > 1024 * 1024 * 1024 {
            return err("Histórico atingiu limite de 1 GiB; arquive o perfil");
        }
        f.write_all(line.as_bytes())
            .and_then(|_| f.sync_all())
            .map_err(ioerr)?;
        self.head = hash;
        self.sequence += 1;
        Ok(())
    }
}
fn safe_profile_path(root: &Path, name: &str) -> Result<PathBuf> {
    if name.is_empty()
        || name.len() > 48
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return err("Perfil: use 1..48 letras ASCII, números, _ ou -");
    }
    fs::create_dir_all(root).map_err(ioerr)?;
    let root = fs::canonicalize(root).map_err(ioerr)?;
    let path = root.join(name);
    if let Ok(meta) = fs::symlink_metadata(&path) {
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return err("Perfil não pode ser link simbólico/arquivo");
        }
    } else {
        fs::create_dir(&path).map_err(ioerr)?;
    }
    if fs::canonicalize(&path).map_err(ioerr)?.parent() != Some(root.as_path()) {
        return err("Perfil fora da raiz");
    }
    Ok(path)
}

struct Profile {
    name: String,
    directory: PathBuf,
    model: NativeSeq2Seq,
    history: EventHistory,
    _lease: Lease,
    replay: VecDeque<Pair>,
    memory: VecDeque<Pair>,
    restricted: bool,
    online: bool,
    last_inspection: Json,
    last_train: Json,
    kernel: Polynomial,
}
impl Profile {
    fn open(root: &Path, name: &str) -> Result<Self> {
        let directory = safe_profile_path(root, name)?;
        for entry in ["events.jsonl", "checkpoint.json"] {
            let path = directory.join(entry);
            if fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
                return err("Arquivo de estado não pode ser link simbólico");
            }
        }
        let lease = Lease::acquire(&directory)?;
        let history = EventHistory::open(directory.join("events.jsonl"))?;
        let seed = u64::from_le_bytes(sha256(name.as_bytes())[..8].try_into().unwrap());
        let mut profile = Self {
            name: name.into(),
            directory,
            model: NativeSeq2Seq::new(16, seed)?,
            history,
            _lease: lease,
            replay: VecDeque::new(),
            memory: VecDeque::new(),
            restricted: true,
            online: true,
            last_inspection: Json::Null,
            last_train: Json::Null,
            kernel: PolyParser::parse("x*x+2*x+1")?,
        };
        let path = profile.directory.join("checkpoint.json");
        if path.exists() {
            let bytes = read_bounded(&path)?;
            let checkpoint =
                Json::parse(std::str::from_utf8(&bytes).map_err(|_| "Checkpoint UTF-8 inválido")?)?;
            let body = checkpoint.get("body");
            if checkpoint.get("sha256").string()? != digest(body.encode().as_bytes()) {
                return err("Checksum de checkpoint inválido");
            }
            if body.get("format").string()? != "neuron-rust-1" {
                return err("Formato incompatível; NPZ/Python não são checkpoints Rust");
            }
            if !profile
                .history
                .contains_anchor(body.get("history_anchor").string()?)?
            {
                return err("Checkpoint se refere a eventos ausentes; histórico truncado/rollback");
            }
            profile.model = NativeSeq2Seq::from_json(body.get("model"))?;
            profile.restricted = body.get("restricted") == &Json::Bool(true);
            profile.online = body.get("online") == &Json::Bool(true);
            for pair in body.get("replay").array()? {
                profile.replay.push_back(Pair::from_json(pair)?);
            }
            for pair in body.get("memory").array()? {
                profile.memory.push_back(Pair::from_json(pair)?);
            }
            if profile.replay.len() > 16 || profile.memory.len() > 1024 {
                return err("Checkpoint excede capacidades");
            }
            profile.kernel = PolyParser::parse(body.get("kernel").string()?)?;
        }
        profile.history.append(
            "session.open",
            obj([("profile", js(name)), ("version", js(VERSION))]),
        )?;
        Ok(profile)
    }
    fn save(&mut self) -> Result<()> {
        let body = obj([
            ("format", js("neuron-rust-1")),
            ("history_anchor", js(&self.history.head)),
            ("model", self.model.to_json()),
            ("restricted", Json::Bool(self.restricted)),
            ("online", Json::Bool(self.online)),
            (
                "replay",
                Json::Arr(self.replay.iter().map(Pair::json).collect()),
            ),
            (
                "memory",
                Json::Arr(self.memory.iter().map(Pair::json).collect()),
            ),
            ("kernel", js(self.kernel.text())),
        ]);
        let checksum = digest(body.encode().as_bytes());
        let checkpoint = obj([("body", body), ("sha256", js(&checksum))]);
        atomic_write(
            &self.directory.join("checkpoint.json"),
            checkpoint.encode().as_bytes(),
        )?;
        self.history.append(
            "checkpoint.saved",
            obj([
                ("sha256", js(checksum)),
                ("model_signature", js(self.model.signature())),
            ]),
        )
    }
    fn remember(&mut self, pair: Pair) {
        self.replay.push_back(pair.clone());
        if self.replay.len() > 16 {
            self.replay.pop_front();
        }
        self.memory.push_back(pair);
        if self.memory.len() > 1024 {
            self.memory.pop_front();
        }
    }
    fn learn(&mut self, pair: &Pair) -> Result<TrainReport> {
        let replay: Vec<_> = self.replay.iter().cloned().collect();
        let before = self.model.clone();
        let report = train_pair(
            &mut self.model,
            pair,
            &replay,
            self.restricted,
            if self.restricted { 0.06 } else { 0.12 },
        )?;
        self.last_train = report.json();
        if let Err(error) = self
            .history
            .append("weights.online", self.last_train.clone())
        {
            self.model = before;
            return Err(error);
        }
        self.remember(pair.clone());
        Ok(report)
    }
    fn reply(&mut self, prompt: &str, peer: bool) -> Result<String> {
        if prompt.len() > 16384 {
            return err("Mensagem excede 16 KiB");
        }
        self.history
            .append(if peer { "peer.input" } else { "user.input" }, js(prompt))?;
        let formal = natural_math(prompt)?;
        let reply = if let Some((operation, args)) = formal {
            let result = calcular(&operation, &args)?;
            let output = humanize(&operation, &args, &result);
            self.last_inspection = obj([
                ("operation", js(&operation)),
                ("args", Json::Arr(args)),
                ("result", result),
            ]);
            if self.online && !peer {
                let pair = Pair {
                    input: prompt.into(),
                    output: output.clone(),
                    trusted: true,
                };
                self.learn(&pair)?;
            }
            output
        } else {
            // O canal de linguagem é sempre o encoder–decoder; recuperação HDC é
            // uma ferramenta explícita (/lembrar), não um LLM alternativo.
            let (output, inspection) = self.model.generate(prompt, &zero_context())?;
            self.last_inspection = inspection;
            if self.online && !peer {
                self.learn(&Pair {
                    input: prompt.into(),
                    output: prompt.into(),
                    trusted: false,
                })?;
            }
            if output.is_empty() {
                "O decoder produziu fim de sequência. Treine o modelo local com /ensinar ou /treinar.".into()
            } else {
                output
            }
        };
        self.history.append(
            if peer {
                "peer.output"
            } else {
                "assistant.output"
            },
            js(&reply),
        )?;
        self.save()?;
        Ok(reply)
    }
    fn recall(&self, query: &str) -> Json {
        let q = semantic_vector(query, 256);
        let mut scores: Vec<_> = self
            .memory
            .iter()
            .map(|p| (dot(&q, &semantic_vector(&p.input, 256)), p))
            .collect();
        scores.sort_by(|a, b| b.0.total_cmp(&a.0));
        Json::Arr(
            scores
                .into_iter()
                .take(5)
                .map(|(score, p)| {
                    obj([("lexical_hdc_similarity", jn(score)), ("example", p.json())])
                })
                .collect(),
        )
    }
    fn train_file(&mut self, path: &Path, epochs: usize, resize: Option<usize>) -> Result<Json> {
        let data = import_dataset(path, 256)?;
        let report = evolve_dataset(
            &mut self.model,
            &data,
            &self.replay.iter().cloned().collect::<Vec<_>>(),
            self.restricted,
            epochs,
            resize,
            30.,
        )?;
        let result = report.json();
        self.history.append(
            "dataset.evolution",
            obj([
                ("path", js(path.to_string_lossy())),
                ("source_sha256", js(digest(&read_bounded(path)?))),
                ("result", result.clone()),
            ]),
        )?;
        self.save()?;
        Ok(result)
    }
    fn self_improve(&mut self, rounds: usize, grow: bool) -> Result<Json> {
        if rounds == 0 || rounds > 100 {
            return err("Auto: escolha 1..100 rodadas limitadas");
        }
        let mut rng = Rng::new(self.model.updates ^ now());
        let mut reports = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(60);
        for round in 0..rounds {
            if Instant::now() > deadline {
                break;
            }
            let mut pairs = Vec::new();
            for _ in 0..12 {
                let a = 2 + rng.index(100) as u64;
                let b = 2 + rng.index(100) as u64;
                let input = format!("Qual é o máximo divisor comum de {a} e {b}?");
                let answer = gcd(a, b);
                let verifier = calcular("nt.gcd", &[exact_int(a as u128), exact_int(b as u128)])?;
                if uint(&verifier)? != answer {
                    return err("Oráculo discordante");
                }
                pairs.push(Pair {
                    input,
                    output: answer.to_string(),
                    trusted: true,
                });
            }
            let resize = if grow && self.model.dimension < 64 && round % 2 == 0 {
                Some((self.model.dimension + 8).min(64))
            } else {
                None
            };
            let report = evolve_dataset(
                &mut self.model,
                &pairs,
                &self.replay.iter().cloned().collect::<Vec<_>>(),
                self.restricted,
                1,
                resize,
                deadline
                    .saturating_duration_since(Instant::now())
                    .as_secs_f64()
                    .min(5.)
                    .max(0.1),
            )?;
            let record = report.json();
            self.history.append("auto.round", record.clone())?;
            reports.push(record);
            self.save()?;
        }
        Ok(Json::Arr(reports))
    }
    fn hot_kernel(&mut self, expression: &str) -> Result<Json> {
        let proof = symbolic_equivalence_check(&self.kernel.text(), expression)?;
        if proof.get("proved") != &Json::Bool(true) {
            return err("Kernel rejeitado: equivalência polinomial não demonstrada");
        }
        let candidate = PolyParser::parse(expression)?;
        let start = Instant::now();
        let mut checksum = 0.;
        for i in 0..1000 {
            let vars = BTreeMap::from([("x".into(), i as f64 / 1000.)]);
            checksum += candidate.eval(&vars)?;
        }
        std::hint::black_box(checksum);
        self.kernel = candidate;
        let report = obj([
            ("proof", proof),
            (
                "activation",
                js("AST polinomial em RAM; não reescreve o binário/fonte Rust"),
            ),
            (
                "sample_runtime_ns",
                js(start.elapsed().as_nanos().to_string()),
            ),
        ]);
        self.history.append("kernel.hot", report.clone())?;
        self.save()?;
        Ok(report)
    }
}

fn natural_math(prompt: &str) -> Result<Option<(String, Vec<Json>)>> {
    let text = prompt.trim().trim_end_matches(['?', '.', '!']);
    let lower = text.to_lowercase();
    if let Some(rest) = lower.strip_prefix("prove que ") {
        for marker in [" é equivalente a ", " equivale a ", " é igual a "] {
            if let Some((a, b)) = rest.split_once(marker) {
                return Ok(Some(("system.equivalence".into(), vec![js(a), js(b)])));
            }
        }
        return err("Use: prove que (a+b)**2 é equivalente a a**2+2*a*b+b**2");
    }
    if lower.contains("fator") || lower.contains("primo") {
        let numbers: Vec<_> = lower
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .collect();
        if numbers.len() == 1 {
            let n: u64 = numbers[0].parse().map_err(|_| "Inteiro excede u64")?;
            return Ok(Some((
                if lower.contains("fator") {
                    "nt.factor"
                } else {
                    "nt.is_prime"
                }
                .into(),
                vec![js(n.to_string())],
            )));
        }
    }
    for prefix in ["calcule ", "calcular ", "quanto é ", "quanto e "] {
        if let Some(expr) = lower.strip_prefix(prefix) {
            if !expr.chars().any(char::is_alphabetic) {
                return Ok(Some(("system.arithmetic".into(), vec![js(expr)])));
            }
        }
    }
    Ok(None)
}
fn humanize(operation: &str, args: &[Json], result: &Json) -> String {
    match operation {
        "nt.factor" => {
            if let Json::Obj(o) = result {
                let mut factors: Vec<_> = o.iter().collect();
                factors.sort_by_key(|(k, _)| k.parse::<u64>().unwrap_or(0));
                let s = factors
                    .iter()
                    .map(|(p, e)| {
                        if e == &&jn(1.) {
                            p.to_string()
                        } else {
                            format!("{}^{}", p, e.encode())
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" × ");
                if o.is_empty() {
                    "1 é a unidade; não possui fatores primos.".into()
                } else {
                    format!("Os fatores primos são {s}.")
                }
            } else {
                result.encode()
            }
        }
        "nt.is_prime" => format!(
            "{} {}.",
            args.first()
                .and_then(|x| x.string().ok())
                .map(str::to_string)
                .unwrap_or_else(|| args[0].encode()),
            if result == &Json::Bool(true) {
                "é primo no domínio u64, verificado por Miller–Rabin determinístico"
            } else {
                "não é primo"
            }
        ),
        "system.arithmetic" => format!("O resultado exato é {}.", result.string().unwrap_or("?")),
        "system.equivalence" => {
            if result.get("proved") == &Json::Bool(true) {
                "Comprovei a equivalência no anel de polinômios sobre Q. Isso não é uma prova de equivalência para aritmética de ponto flutuante.".into()
            } else {
                format!(
                    "As expressões não são polinômios equivalentes. Diferença: {}.",
                    result.get("difference").string().unwrap_or("?")
                )
            }
        }
        _ => result.encode(),
    }
}
fn coverage() -> Json {
    obj([
    ("version",js(VERSION)),("complete_python_translation",Json::Bool(false)),("python_interpreter_used",Json::Bool(false)),("build",js("rustc --edition=2021 -O rust/neuron.rs -o neuron")),
    ("native",Json::Arr([
        "encoder/decoder RNN, atenção cruzada, BPTT e ciclos latentes",
        "DendriticCore classificação, gradientes analíticos e divisão de roteamento",
        "TCP/Hildreth, restrições constitucionais numéricas, replay e retenção",
        "perfis isolados, treino online, checkpoints JSON e histórico SHA256",
        "JSON/JSONL/NDJSON/CSV/TSV/TXT/MD, holdout agrupado, treino e resize candidato",
        "geração sintética verificada, ciclo limitado, promoção/rollback em memória",
        "provas Q[x] i128, derivada polinomial, hot swap de expressão interpretada",
        "HDC lexical local, álgebra VSA bipolar, tokenizer BPE independente",
        "benchmark local sem treino/solver; inspeção de tensores e HTTP autenticado",
        "conversa entre perfis no mesmo processo; saídas dos pares não treinam pesos"
    ].iter().map(|s|js(*s)).collect())),
    ("missing_from_python",Json::Arr([
        "Torch/Transformers e download/hot-swap de backbones Hugging Face",
        "Arrow, Feather, Parquet, PDF, DOCX, XLSX, HTML/XML e download HF",
        "catálogo OMNIMAT completo de 424 operações, Gröbner, categorias e JIT x86-64",
        "WorldModel/CognitiveUnit completos, MPC, afetos, sonho, metas, ArmsRace original",
        "CompressedLanguageModel PPM e uso do BPE como tokenizer neural",
        "SQLite WAL, NPZ e compatibilidade de checkpoints Python",
        "federação autenticada, P2P, desktop, sensores e ToolHarness do Python",
        "download/leaderboards MMLU/ARC/GSM8K, avaliação HumanEval",
        "sandbox de bytecode geral, alteração/recompilação automática de fonte Rust",
        "WebSocket e observatório WebGL original, daemon de autotreino contínuo"
    ].iter().map(|s|js(*s)).collect())),
    ("native_math_operations",Json::Arr(MATH_OPS.iter().map(|s|js(*s)).collect())),
    ("checkpoint_compatibility",js("Formato neuron-rust-1; não lê SQLite/NPZ/Python; diretórios separados por padrão.")),
    ("scientific_limits",js("Sem garantia de fluência, ausência de esquecimento, progresso ou alinhamento; recorrência latente não é prova de raciocínio. Não é AGI. Histórico/checkpoint em arquivos distintos, não uma transação SQLite."))
])
}

fn local_benchmark(model: &NativeSeq2Seq, path: &Path, limit: usize) -> Result<Json> {
    if limit == 0 || limit > 256 {
        return err("Benchmark: limite 1..256");
    }
    let data = read_bounded(path)?;
    let text = std::str::from_utf8(&data).map_err(|_| "Benchmark deve ser JSONL UTF-8")?;
    let signature = model.signature();
    let mut results = Vec::new();
    let mut correct = 0;
    let mut token_loss = 0.;
    for line in text.lines().filter(|s| !s.trim().is_empty()).take(limit) {
        let j = Json::parse(line)?;
        let prompt = j
            .get("question")
            .string()
            .or_else(|_| j.get("input").string())?;
        if let Ok(choices) = j.get("choices").array() {
            if choices.len() < 2 || choices.len() > 16 {
                return err("Benchmark: 2..16 escolhas");
            }
            let target = j.get("answer_index").usize()?;
            if target >= choices.len() {
                return err("Benchmark: answer_index inválido");
            }
            let mut scores = Vec::new();
            for choice in choices {
                let answer = choice.string()?;
                let (loss, _) = model.loss_grad(prompt, &zero_context(), answer)?;
                let count = model.tokens(answer, model.config.output_tokens - 1).len() + 1;
                scores.push(-loss * count as f64);
            }
            let predicted = (0..scores.len())
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                .unwrap();
            if predicted == target {
                correct += 1;
            }
            results.push(obj([
                ("predicted", jn(predicted as f64)),
                ("target", jn(target as f64)),
                ("conditional_log_likelihood", ja(&scores)),
            ]));
        } else {
            let answer = j
                .get("answer")
                .string()
                .or_else(|_| j.get("output").string())?;
            let (generated, _) = model.generate(prompt, &zero_context())?;
            let (loss, _) = model.loss_grad(prompt, &zero_context(), answer)?;
            token_loss += loss;
            if generated.trim() == answer.trim() {
                correct += 1;
            }
            results.push(obj([
                ("generated", js(generated)),
                ("target", js(answer)),
                ("mean_token_nll", jn(loss)),
            ]));
        }
    }
    if results.is_empty() {
        return err("Benchmark vazio");
    }
    if model.signature() != signature {
        return err("Benchmark alterou pesos");
    }
    let protocol = obj([
        ("version", js("rust-local-v1")),
        ("model_signature", js(&signature)),
        ("dataset_sha256", js(digest(&data))),
        ("limit", jn(limit as f64)),
        ("input_tokens", jn(model.config.input_tokens as f64)),
        ("output_tokens", jn(model.config.output_tokens as f64)),
        ("thought_budget", jn(model.config.thought_budget as f64)),
        ("solver_used", Json::Bool(false)),
        ("training_enabled", Json::Bool(false)),
        (
            "choice_score",
            js("sum target-token log probabilities including EOS; no length normalization"),
        ),
    ]);
    Ok(obj([
        ("protocol_id", js(digest(protocol.encode().as_bytes()))),
        ("protocol", protocol),
        ("count", jn(results.len() as f64)),
        ("accuracy", jn(correct as f64 / results.len() as f64)),
        ("summed_example_mean_nll", jn(token_loss)),
        ("results", Json::Arr(results)),
        (
            "comparison",
            js("Resultados locais; não equivalem a scores canônicos ou referências externas."),
        ),
    ]))
}

struct Profiles {
    root: PathBuf,
    current: String,
    opened: BTreeMap<String, Profile>,
}
impl Profiles {
    fn new(root: PathBuf, name: &str) -> Result<Self> {
        let profile = Profile::open(&root, name)?;
        Ok(Self {
            root,
            current: name.into(),
            opened: BTreeMap::from([(name.into(), profile)]),
        })
    }
    fn switch(&mut self, name: &str) -> Result<()> {
        if !self.opened.contains_key(name) {
            if self.opened.len() >= 8 {
                return err("No máximo oito perfis simultâneos");
            }
            let p = Profile::open(&self.root, name)?;
            self.opened.insert(name.into(), p);
        }
        self.current = name.into();
        Ok(())
    }
    fn profile(&mut self) -> &mut Profile {
        self.opened.get_mut(&self.current).unwrap()
    }
    fn state(&self) -> Json {
        let p = &self.opened[&self.current];
        obj([
            ("version", js(VERSION)),
            ("profile", js(&p.name)),
            ("restricted", Json::Bool(p.restricted)),
            ("online", Json::Bool(p.online)),
            ("dimension", jn(p.model.dimension as f64)),
            (
                "parameters",
                jn(p.model.p.values().map(Vec::len).sum::<usize>() as f64),
            ),
            ("updates", js(p.model.updates.to_string())),
            ("model_signature", js(p.model.signature())),
            ("thought_budget", jn(p.model.config.thought_budget as f64)),
            ("history_events", js(p.history.sequence.to_string())),
            ("last_train", p.last_train.clone()),
            ("inspection", p.last_inspection.clone()),
        ])
    }
    fn dialogue(&mut self, a: &str, b: &str, topic: &str) -> Result<String> {
        if a == b {
            return err("Escolha dois perfis distintos");
        }
        let previous = self.current.clone();
        self.switch(a)?;
        self.switch(b)?;
        self.current = previous;
        let first = self.opened.get_mut(a).unwrap().reply(topic, true)?;
        let second = self
            .opened
            .get_mut(b)
            .unwrap()
            .reply(&format!("{a}: {first}"), true)?;
        Ok(format!("{a} > {first}\n{b} > {second}\n(Mensagens dos pares identificadas; sem treino automático.)"))
    }
    fn save_all(&mut self) -> Result<()> {
        for p in self.opened.values_mut() {
            p.save()?;
        }
        Ok(())
    }
}
const HELP: &str = r#"NEURON Rust — encoder/decoder local; port incremental (./neuron --coverage).
Uso: ./neuron [agent] [--root DIRETORIO] [--profile NOME]
     ./neuron --menu | --coverage | --catalog | --version
     ./neuron calc OPERACAO '[ARGUMENTOS JSON]'
     ./neuron demo [PASSOS] | bpe TEXTO
     ./neuron prove 'EXPR1' 'EXPR2'
     ./neuron fixture ARQUIVO.json  (equações/gradientes para verificação)
No chat:
 /ajuda                         lista comandos
 /menu                          menu do laboratório
 /estado | /inspecionar          pesos, esforço e tensores observados
 /esforco N [SEGUNDOS]           0..256, total <=8192 recorrências
 /esforco baixo|medio|alto|maximo|+   2,8,16,64 ou dobra
 /modo restrito|irrestrito       constituição numérica/plasticidade
 /aprender ligar|desligar        atualização real por turno
 /ensinar pergunta || resposta   supervisão explícita e treino TCP
 /treinar CAMINHO [EPOCAS]       JSON/JSONL/CSV/TSV/TXT/MD; holdout
 /crescer N CAMINHO              arquitetura candidata, avaliada antes de promover
 /podar N CAMINHO                dimensão alvo, avaliada antes de promover
 /auto RODADAS [crescer]         1..100, prazo total, síntese e holdout
 /perfil NOME                   cria/abre/muda; até oito residentes
 /perfis                        lista residentes isolados
 /conversar ALICE BOB TEMA       troca entre dois perfis sem supervisão circular
 /lembrar TEXTO                 consulta HDC lexical (não memória infinita)
 /calc OPERACAO [JSON]           operações nativas; /catalogo lista
 /provar EXPR1 || EXPR2          equivalência racional polinomial limitada
 /kernel EXPRESSAO              hot swap AST equivalente, sem recompilar Rust
 /benchmark CAMINHO [LIMITE]     JSONL local, sem treino nem solver
 /salvar                        checkpoint e histórico
 /painel                        informa comando de servidor HTTP
 /cobertura                     lista recursos portados e ausentes
 sair                           salva e encerra
Servidor: ./neuron serve --port 8765 [--root ...] [--profile ...]
Datasets e caminhos: use aspas para espaços. Nenhuma instalação/download implícito.
Pesos iniciais aleatórios: treinar um modelo pequeno não o torna fluente como T5.
"#;
fn words(text: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut quote = None;
    for c in text.chars() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
            } else {
                buf.push(c);
            }
        } else if c == '\'' || c == '"' {
            quote = Some(c);
        } else if c.is_whitespace() {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
        } else {
            buf.push(c);
        }
    }
    if quote.is_some() {
        return err("Aspas não fechadas");
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    Ok(out)
}
fn parse_count(s: &str) -> Result<usize> {
    s.parse()
        .map_err(|_| "Inteiro não negativo esperado".into())
}
fn command(app: &mut Profiles, text: &str) -> Result<String> {
    let (command, rest) = text.split_once(' ').unwrap_or((text, ""));
    let rest = rest.trim();
    match command{
        "/ajuda"|"/help"=>Ok(HELP.into()),"/menu"=>Ok("1 Conversar  2 Estado (/estado)  3 Treinar (/treinar)\n4 Matemática (/calc)  5 Perfis (/perfil)  6 Auto (/auto)\n7 Esforço (/esforco)  8 Benchmark (/benchmark)  9 Cobertura (/cobertura)\nOs modos são selecionados pelos comandos entre parênteses.".into()),
        "/estado"=>Ok(app.state().encode()),"/inspecionar"=>Ok(app.profile().last_inspection.encode()),"/cobertura"=>Ok(coverage().encode()),"/catalogo"=>Ok(MATH_OPS.join("\n")),
        "/esforco"=>{let w=words(rest)?;let p=app.profile();let first=w.first().ok_or("Use /esforco N [segundos]")?;let n=match first.as_str(){"baixo"=>2,"medio"|"médio"=>8,"alto"=>16,"maximo"|"máximo"=>64,"+"=>(p.model.config.thought_budget.max(1)*2).min(256),s=>parse_count(s)?};let mut config=p.model.config.clone();config.thought_budget=n;if let Some(sec)=w.get(1){config.inference_seconds=sec.parse().map_err(|_|"Prazo inválido")?;}config.validate()?;p.model.config=config;p.history.append("config.effort",jn(n as f64))?;p.save()?;Ok(format!("Orçamento: {n} recorrências latentes por token, prazo {} s. O teto de tokens pode diminuir para respeitar 8192 recorrências. Isto não garante qualidade de raciocínio.",p.model.config.inference_seconds))},
        "/modo"=>{let p=app.profile();p.restricted=match rest{"restrito"=>true,"irrestrito"=>false,_=>return err("Use /modo restrito ou /modo irrestrito")};p.history.append("config.mode",js(rest))?;p.save()?;Ok(format!("Modo {rest}; permissões de sistema e limites de recursos permanecem explícitos."))},
        "/aprender"=>{let p=app.profile();p.online=match rest{"ligar"|"on"=>true,"desligar"|"off"=>false,_=>return err("Use /aprender ligar|desligar")};p.save()?;Ok(format!("Aprendizado online {}.",if p.online{"ligado"}else{"desligado"}))},
        "/ensinar"=>{let(a,b)=rest.split_once("||").ok_or("Use /ensinar pergunta || resposta")?;let pair=Pair{input:a.trim().into(),output:b.trim().into(),trusted:true};validate_pairs(vec![pair.clone()])?;let p=app.profile();let report=p.learn(&pair)?;p.save()?;Ok(report.json().encode())},
        "/treinar"=>{let w=words(rest)?;let path=w.first().ok_or("Use /treinar CAMINHO [epocas]")?;let epochs=if let Some(s)=w.get(1){parse_count(s)?}else{1};Ok(app.profile().train_file(Path::new(path),epochs,None)?.encode())},
        "/crescer"|"/podar"=>{let w=words(rest)?;if w.len()!=2{return err("Use /crescer N CAMINHO ou /podar N CAMINHO");}let n=parse_count(&w[0])?;let p=app.profile();if(command=="/crescer"&&n<=p.model.dimension)||(command=="/podar"&&n>=p.model.dimension){return err("A dimensão alvo não corresponde à direção escolhida");}Ok(p.train_file(Path::new(&w[1]),1,Some(n))?.encode())},
        "/auto"=>{let w=words(rest)?;let rounds=parse_count(w.first().ok_or("Use /auto RODADAS [crescer]")?)?;Ok(app.profile().self_improve(rounds,w.get(1).is_some_and(|s|s=="crescer"))?.encode())},
        "/perfil"=>{app.switch(rest)?;Ok(format!("Perfil ativo: {}",app.current))},"/perfis"=>Ok(app.opened.keys().cloned().collect::<Vec<_>>().join(", ")),
        "/conversar"=>{let mut w=rest.splitn(3,' ');let a=w.next().ok_or("Falta perfil A")?;let b=w.next().ok_or("Falta perfil B")?;let topic=w.next().ok_or("Falta tema")?;app.dialogue(a,b,topic)},
        "/lembrar"=>Ok(app.profile().recall(rest).encode()),
        "/calc"=>{let(op,args)=rest.split_once(' ').ok_or("Use /calc operacao [JSON]")?;let args=Json::parse(args)?;let result=calcular(op,args.array()?)?;app.profile().history.append("math.explicit",obj([("operation",js(op)),("args",args.clone()),("result",result.clone())]))?;Ok(humanize(op,args.array()?,&result))},
        "/provar"=>{let(a,b)=rest.split_once("||").ok_or("Use /provar EXPR1 || EXPR2")?;let result=symbolic_equivalence_check(a.trim(),b.trim())?;app.profile().history.append("math.proof",result.clone())?;Ok(result.encode())},
        "/kernel"=>Ok(app.profile().hot_kernel(rest)?.encode()),
        "/benchmark"=>{let w=words(rest)?;let path=w.first().ok_or("Use /benchmark CAMINHO [limite]")?;let limit=if let Some(s)=w.get(1){parse_count(s)?}else{32};let p=app.profile();let result=local_benchmark(&p.model,Path::new(path),limit)?;p.history.append("benchmark.local",result.clone())?;atomic_write(&p.directory.join("benchmark.json"),result.encode().as_bytes())?;Ok(result.encode())},
        "/salvar"=>{app.save_all()?;Ok("Perfis salvos.".into())},"/painel"=>Ok(format!("Para iniciar um servidor próprio: ./neuron serve --root {} --profile {} --port 8765. Encerre esta sessão antes de abrir o mesmo perfil em outro processo.",app.root.display(),app.current)),
        s if s.starts_with('/')=>err("Comando desconhecido. Use /ajuda; nenhuma ação foi executada."),_=>app.profile().reply(text,false)
    }
}

const HTML: &str = r##"<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>NEURON Rust</title><style>body{background:#101620;color:#dbebff;font:16px system-ui;max-width:1000px;margin:24px auto;padding:16px}canvas{width:100%;height:220px;background:#080d14;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto}input{width:75%;padding:12px}button{padding:12px}small{color:#aabbcc}</style><h1>NEURON · Rust nativo</h1><small>Port incremental. Ativações observadas do encoder–decoder; não são pensamentos em linguagem natural.</small><canvas id="brain" width="900" height="220"></canvas><p id="status"></p><pre id="chat"></pre><form id="form"><input id="prompt" maxlength="16384" placeholder="Mensagem ou /ajuda" autocomplete="off"><button>Enviar</button></form><details><summary>Tensores e diagnóstico</summary><pre id="state"></pre></details><script>
'use strict';const token=location.hash.slice(1);history.replaceState(null,'',location.pathname);const by=id=>document.getElementById(id);let busy=false;let state={};async function request(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'X-Neuron-Token':token,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const j=await r.json();if(!r.ok)throw Error(j.error||r.status);return j;}function draw(){const c=by('brain'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);const a=(state.inspection||{}).decoder_states||[];const v=a.flat().slice(-400);v.forEach((n,i)=>{const xx=18+(i%50)*17,yy=20+Math.floor(i/50)*25;x.fillStyle=n>=0?'#51dbe0':'#e87da0';x.globalAlpha=.2+.8*Math.min(1,Math.abs(n));x.beginPath();x.arc(xx,yy,2+5*Math.abs(n),0,Math.PI*2);x.fill();});x.globalAlpha=1;x.fillStyle='#bcd';x.fillText(v.length?'Ativações reais; escala |h|':'Envie uma mensagem para capturar ativações.',15,210);}async function refresh(){if(busy)return;try{state=await request('/state');by('state').textContent=JSON.stringify(state,null,2);by('status').textContent=`Perfil: ${state.profile} | dimensão ${state.dimension} | atualizações ${state.updates} | esforço ${state.thought_budget}`;draw();}catch(e){by('status').textContent=e.message;}}by('form').onsubmit=async e=>{e.preventDefault();if(busy)return;const p=by('prompt').value;if(!p.trim())return;busy=true;by('chat').textContent+='Você > '+p+'\n';by('prompt').value='';try{const j=await request('/chat',{prompt:p});by('chat').textContent+='NEURON > '+j.reply+'\n\n';}catch(e){by('chat').textContent+='Erro: '+e.message+'\n';}finally{busy=false;refresh();}};refresh();setInterval(refresh,1500);
</script></html>"##;
fn secure_token() -> Result<String> {
    #[cfg(unix)]
    {
        let mut bytes = [0u8; 32];
        File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut bytes))
            .map_err(ioerr)?;
        Ok(hex(&bytes))
    }
    #[cfg(windows)]
    {
        #[link(name = "bcrypt")]
        unsafe extern "system" {
            fn BCryptGenRandom(h: *mut std::ffi::c_void, b: *mut u8, n: u32, flags: u32) -> i32;
        }
        let mut bytes = [0u8; 32];
        if unsafe { BCryptGenRandom(std::ptr::null_mut(), bytes.as_mut_ptr(), 32, 2) } != 0 {
            return err("BCryptGenRandom falhou");
        }
        Ok(hex(&bytes))
    }
    #[cfg(not(any(unix, windows)))]
    {
        err("Gerador seguro não disponível nesta plataforma")
    }
}
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b) {
        result |= x ^ y;
    }
    result == 0
}
fn http_response(stream: &mut TcpStream, status: u16, kind: &str, body: &str) -> Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let header=format!("HTTP/1.1 {status} {reason}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Security-Policy: default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'\r\n\r\n",body.len());
    stream
        .write_all(header.as_bytes())
        .and_then(|_| stream.write_all(body.as_bytes()))
        .map_err(ioerr)
}
fn handle_http(
    mut stream: TcpStream,
    app: &Arc<Mutex<Profiles>>,
    token: &str,
    port: u16,
) -> Result<()> {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .map_err(ioerr)?;
    stream
        .set_write_timeout(Some(Duration::from_secs(3)))
        .map_err(ioerr)?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    let header_end = loop {
        if buf.len() > 16384 {
            return http_response(&mut stream, 400, "text/plain", "Cabeçalho excedido");
        }
        let n = stream.read(&mut chunk).map_err(ioerr)?;
        if n == 0 {
            return err("Requisição incompleta");
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|b| b == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let headers = std::str::from_utf8(&buf[..header_end]).map_err(|_| "Cabeçalho inválido")?;
    let mut lines = headers.split("\r\n");
    let first = lines.next().ok_or("Requisição ausente")?;
    let request: Vec<_> = first.split_whitespace().collect();
    if request.len() != 3 {
        return err("Requisição inválida");
    }
    let (method, path) = (request[0].to_string(), request[1].to_string());
    let mut h = BTreeMap::new();
    for line in lines.filter(|s| !s.is_empty()) {
        let (k, v) = line.split_once(':').ok_or("Cabeçalho inválido")?;
        if h.insert(k.to_ascii_lowercase(), v.trim().to_string())
            .is_some()
        {
            return http_response(&mut stream, 400, "text/plain", "Cabeçalho duplicado");
        }
    }
    let allowed_host = format!("127.0.0.1:{port}");
    if h.get("host") != Some(&allowed_host) {
        return http_response(
            &mut stream,
            403,
            "text/plain",
            "Host inválido; use 127.0.0.1",
        );
    }
    if h.contains_key("transfer-encoding") {
        return http_response(
            &mut stream,
            400,
            "text/plain",
            "Transfer-Encoding não suportado",
        );
    }
    if let Some(origin) = h.get("origin") {
        if origin != &format!("http://127.0.0.1:{port}") {
            return http_response(&mut stream, 403, "text/plain", "Origin inválida");
        }
    }
    if method == "GET" && path == "/" {
        return http_response(&mut stream, 200, "text/html; charset=utf-8", HTML);
    }
    if !constant_time_eq(
        h.get("x-neuron-token")
            .map_or(b"".as_slice(), String::as_bytes),
        token.as_bytes(),
    ) {
        return http_response(
            &mut stream,
            403,
            "application/json",
            "{\"error\":\"Token inválido\"}",
        );
    }
    let result = if method == "GET" && path == "/state" {
        app.lock().map_err(|_| "Estado indisponível")?.state()
    } else if method == "POST" && path == "/chat" {
        let length = h
            .get("content-length")
            .ok_or("Falta Content-Length")?
            .parse::<usize>()
            .map_err(|_| "Content-Length inválido")?;
        if length > 65536 {
            return http_response(&mut stream, 400, "text/plain", "Mensagem excedida");
        }
        while buf.len() < header_end + length {
            let n = stream.read(&mut chunk).map_err(ioerr)?;
            if n == 0 {
                return err("Corpo incompleto");
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        let body = Json::parse(
            std::str::from_utf8(&buf[header_end..header_end + length])
                .map_err(|_| "UTF-8 inválido")?,
        )?;
        let mut app = app.lock().map_err(|_| "Estado indisponível")?;
        match command(&mut app, body.get("prompt").string()?) {
            Ok(reply) => obj([("reply", js(reply))]),
            Err(error) => {
                return http_response(
                    &mut stream,
                    400,
                    "application/json",
                    &obj([("error", js(error))]).encode(),
                )
            }
        }
    } else {
        return http_response(
            &mut stream,
            404,
            "application/json",
            "{\"error\":\"Rota desconhecida\"}",
        );
    };
    http_response(
        &mut stream,
        200,
        "application/json; charset=utf-8",
        &result.encode(),
    )
}
fn serve(app: Profiles, port: u16) -> Result<()> {
    if port == 0 {
        return err("Porta não pode ser zero");
    }
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(ioerr)?;
    let token = secure_token()?;
    println!("Observatório HTTP: http://127.0.0.1:{port}/#{token}\nAcesso local autenticado. Ctrl+C encerra; cada operação salva seu estado.");
    let app = Arc::new(Mutex::new(app));
    for incoming in listener.incoming() {
        let stream = incoming.map_err(ioerr)?;
        if let Err(error) = handle_http(stream, &app, &token, port) {
            eprintln!("HTTP: {error}");
        }
    }
    Ok(())
}

// Fixture pública para comparação das equações com a implementação Python.
// Não desserializa objetos executáveis; arrays são JSON e o esquema é validado.
fn fixture(j: &Json) -> Result<Json> {
    match j.get("operation").string()? {
        "seq2seq" => {
            let model = NativeSeq2Seq::from_json(j.get("model"))?;
            let context = numeric_vec(j.get("context"))?;
            let (loss, g) = model.loss_grad(
                j.get("input").string()?,
                &context,
                j.get("answer").string()?,
            )?;
            let (text, inspection) = model.generate(j.get("input").string()?, &context)?;
            Ok(obj([
                ("loss", jn(loss)),
                ("gradient", params_json(&g)),
                ("text", js(text)),
                ("inspection", inspection),
            ]))
        }
        "dendritic" => {
            let mut core = DendriticCore::new(
                j.get("features").usize()?,
                j.get("branches").usize()?,
                j.get("outputs").usize()?,
                j.get("max_experts").usize()?,
                0,
            )?;
            let p = params_from_json(j.get("parameters"))?;
            if p.len() != core.p.len()
                || core
                    .p
                    .iter()
                    .any(|(k, v)| p.get(k).is_none_or(|a| a.len() != v.len()))
            {
                return err("Fixture dendrítica incompatível");
            }
            core.p = p;
            core.active = j.get("active").usize()?;
            if core.active == 0 || core.active > core.max_experts {
                return err("Especialistas inválidos");
            }
            let u = numeric_vec(j.get("input"))?;
            let (loss, g) = core.gradients(&u, j.get("target").usize()?)?;
            Ok(obj([
                ("loss", jn(loss)),
                ("output", ja(&core.forward(&u)?.output)),
                ("gradient", params_json(&g)),
            ]))
        }
        "tcp" => {
            let free = numeric_vec(j.get("free"))?;
            let memories = j
                .get("memories")
                .array()?
                .iter()
                .map(numeric_vec)
                .collect::<Result<Vec<_>>>()?;
            let result = transport_displacement(
                &free,
                &memories,
                &numeric_vec(j.get("importance"))?,
                j.get("strength").number()?,
            )?;
            Ok(obj([
                ("delta", ja(&result.delta)),
                ("fallback", Json::Bool(result.fallback)),
                ("iterations", jn(result.iterations as f64)),
                ("max_violation", jn(result.max_violation)),
            ]))
        }
        _ => err("Fixture desconhecida"),
    }
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("NEURON: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--help" | "-h") => {
            print!("{HELP}");
            return Ok(());
        }
        Some("--coverage") => {
            println!("{}", coverage().encode());
            return Ok(());
        }
        Some("--version") => {
            println!("{VERSION}");
            return Ok(());
        }
        Some("demo") => {
            let steps = if let Some(s) = args.get(1) {
                parse_count(s)?
            } else {
                1000
            };
            println!("{}", supervised_demo(steps)?.encode());
            return Ok(());
        }
        Some("bpe") => {
            let text = args.get(1).ok_or("Use neuron bpe TEXTO")?;
            let mut bpe = BytePairEncoding::new();
            bpe.train(text, 32)?;
            let tokens = bpe.encode(text);
            println!(
                "{}",
                obj([
                    (
                        "token_ids",
                        Json::Arr(tokens.iter().map(|n| jn(*n as f64)).collect())
                    ),
                    ("decoded", js(bpe.decode(&tokens)?)),
                    ("merges", jn(bpe.merges.len() as f64))
                ])
                .encode()
            );
            return Ok(());
        }
        Some("--catalog") => {
            println!("{}", MATH_OPS.join("\n"));
            return Ok(());
        }
        Some("calc") => {
            if args.len() != 3 {
                return err("Use: neuron calc OPERACAO '[ARGUMENTOS JSON]'");
            }
            let j = Json::parse(&args[2])?;
            println!("{}", calcular(&args[1], j.array()?)?.encode());
            return Ok(());
        }
        Some("prove") => {
            if args.len() != 3 {
                return err("Use: neuron prove EXPR1 EXPR2");
            }
            println!(
                "{}",
                symbolic_equivalence_check(&args[1], &args[2])?.encode()
            );
            return Ok(());
        }
        Some("fixture") => {
            if args.len() != 2 {
                return err("Use: neuron fixture ARQUIVO.json");
            }
            let data = read_bounded(Path::new(&args[1]))?;
            println!(
                "{}",
                fixture(&Json::parse(
                    std::str::from_utf8(&data).map_err(|_| "Fixture UTF-8 inválida")?
                )?)?
                .encode()
            );
            return Ok(());
        }
        _ => {}
    }
    let mut root = PathBuf::from("neuron_rust_data/profiles");
    let mut name = "NEURON-1".to_string();
    let mut port = 8765u16;
    let mut i = usize::from(matches!(
        args.first().map(String::as_str),
        Some("agent" | "serve" | "--menu")
    ));
    while i < args.len() {
        let key = &args[i];
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("Falta valor para {key}"))?;
        match key.as_str() {
            "--root" => root = value.into(),
            "--profile" => name = value.into(),
            "--port" => port = value.parse().map_err(|_| "Porta inválida")?,
            _ => return err(format!("Argumento desconhecido: {key}")),
        }
        i += 2;
    }
    let mut app = Profiles::new(root, &name)?;
    if args.first().is_some_and(|x| x == "serve") {
        return serve(app, port);
    }
    println!("NEURON Rust · {VERSION}\nEncoder–decoder nativo com pesos locais. Port incremental: /cobertura.\nPerfil {}. Digite /ajuda, /menu ou sair.\n",app.current);
    if args.first().is_some_and(|x| x == "--menu") {
        println!("{}", command(&mut app, "/menu")?);
    }
    let stdin = io::stdin();
    loop {
        print!("Você > ");
        io::stdout().flush().map_err(ioerr)?;
        let mut line = String::new();
        let n = stdin
            .lock()
            .take(65536)
            .read_line(&mut line)
            .map_err(ioerr)?;
        if n == 0 {
            break;
        }
        if line.len() >= 65536 && !line.ends_with('\n') {
            return err("Entrada excede 64 KiB");
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if ["sair", "exit", "quit", "/sair"].contains(&text.to_lowercase().as_str()) {
            break;
        }
        match command(&mut app, text) {
            Ok(reply) => println!("\nNEURON > {reply}\n"),
            Err(error) => println!("\nNEURON > {error}\n"),
        }
    }
    app.save_all()?;
    println!("Memória e pesos salvos. Até logo.");
    Ok(())
}

// Demonstração supervisionada nativa: Adam + importância diagonal + replay TCP.
// Não substitui os modos cognitivos/ArmsRace completos do Python.
struct PlasticOptimizer {
    m: Vec<f64>,
    v: Vec<f64>,
    importance: Vec<f64>,
    anchor: Vec<f64>,
    step: u64,
}
impl PlasticOptimizer {
    fn new(p: &Params) -> Self {
        let w = flatten(p);
        Self {
            m: vec![0.; w.len()],
            v: vec![0.; w.len()],
            importance: vec![0.; w.len()],
            anchor: w,
            step: 0,
        }
    }
    fn update(&mut self, p: &mut Params, g: &Params, references: &[Vec<f64>]) -> Result<f64> {
        let raw = flatten(g);
        let w = flatten(p);
        if raw.len() != w.len() || self.m.len() != w.len() {
            return err("Adam: parâmetros incompatíveis");
        }
        let mean = self.importance.iter().sum::<f64>() / w.len().max(1) as f64;
        let combined: Vec<_> = raw
            .iter()
            .enumerate()
            .map(|(i, g)| {
                g + 0.01 * (self.importance[i] / mean.max(1e-6)).min(10.) * (w[i] - self.anchor[i])
            })
            .collect();
        let scale = (norm(&combined) / 2.).max(1.);
        self.step += 1;
        let mut free = Vec::with_capacity(w.len());
        for i in 0..w.len() {
            let g = combined[i] / scale;
            self.m[i] = 0.9 * self.m[i] + 0.1 * g;
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g;
            let m = self.m[i] / (1. - 0.9f64.powf(self.step as f64));
            let v = self.v[i] / (1. - 0.999f64.powf(self.step as f64));
            free.push(-0.01 * m / (v.sqrt() + 1e-8));
            self.importance[i] = 0.99 * self.importance[i] + 0.01 * raw[i] * raw[i];
        }
        let result = transport_displacement(&free, references, &self.importance, 2.)?;
        let norm = norm(&result.delta);
        let factor = (0.05 / norm.max(EPS)).min(1.);
        let delta: Vec<_> = result.delta.iter().map(|x| factor * x).collect();
        if w.iter().zip(&delta).any(|(a, b)| !(a + b).is_finite()) {
            return err("Adam: candidato não finito");
        }
        apply_flat(p, &delta)?;
        Ok(norm * factor)
    }
}
fn supervised_demo(steps: usize) -> Result<Json> {
    if steps == 0 || steps > 100000 {
        return err("Demo: passos 1..100000");
    }
    let mut rng = Rng::new(42);
    let mut core = DendriticCore::new(6, 4, 2, 4, 42)?;
    let mut optimizer = PlasticOptimizer::new(&core.p);
    let mut memory = Reservoir::new(64, 11);
    let mut drift = PageHinkley::new(4.);
    let mut correct = 0;
    let mut total_loss = 0.;
    let mut growths = 0;
    for t in 0..steps {
        let x = rng.normal();
        let y = rng.normal();
        let u = vec![x, y, 0., 0., x.sin(), y.cos()];
        let target = usize::from(if t < steps / 2 {
            x * y > 0.
        } else {
            x + y > 0.
        });
        let output = core.forward(&u)?.output;
        if (output[1] > output[0]) == (target == 1) {
            correct += 1;
        }
        let (loss, g) = core.gradients(&u, target)?;
        total_loss += loss;
        let refs = memory
            .items
            .iter()
            .rev()
            .take(4)
            .map(|(u, target): &(Vec<f64>, usize)| {
                core.gradients(u, *target).map(|(_, g)| flatten(&g))
            })
            .collect::<Result<Vec<_>>>()?;
        optimizer.update(&mut core.p, &g, &refs)?;
        memory.push((u, target));
        if drift.update(loss) && core.grow(0)? {
            growths += 1;
            optimizer = PlasticOptimizer::new(&core.p);
        }
    }
    Ok(obj([
        ("mode", js("classification-demo")),
        ("steps", jn(steps as f64)),
        ("prequential_accuracy", jn(correct as f64 / steps as f64)),
        ("mean_loss", jn(total_loss / steps as f64)),
        ("active_experts", jn(core.active as f64)),
        ("growths", jn(growths as f64)),
        ("reservoir_records", jn(memory.items.len() as f64)),
        (
            "limitations",
            js("Demo nativo; resultados não são equivalentes ao benchmark Python original."),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp {
        root: PathBuf,
    }
    impl Temp {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir()
                .join(format!("neuron-rust-test-{}-{nonce}", std::process::id()));
            fs::create_dir_all(&root).unwrap();
            Self { root }
        }
        fn file(&self, name: &str, content: &str) -> PathBuf {
            let p = self.root.join(name);
            fs::write(&p, content).unwrap();
            p
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
    fn close(a: f64, b: f64, tolerance: f64) {
        assert!((a - b).abs() <= tolerance, "{a} != {b}");
    }
    fn tiny() -> NativeSeq2Seq {
        let mut m = NativeSeq2Seq::new(8, 42).unwrap();
        m.config.input_tokens = 16;
        m.config.output_tokens = 6;
        m
    }
    fn pair(a: &str, b: &str) -> Pair {
        Pair {
            input: a.into(),
            output: b.into(),
            trusted: true,
        }
    }
    #[test]
    fn json_unicode_roundtrip() {
        let j = obj([
            ("x", js("á漢字😀\n\t\"\\\u{1}")),
            (
                "array",
                Json::Arr(vec![jn(-3.25), Json::Bool(true), Json::Null]),
            ),
        ]);
        assert_eq!(Json::parse(&j.encode()).unwrap(), j);
        assert_eq!(Json::parse("\"\\ud83d\\ude00\"").unwrap(), js("😀"));
    }
    #[test]
    fn json_invalid_inputs() {
        for s in [
            "NaN",
            "01",
            "1.",
            "1e",
            "-",
            "[1,]",
            "{\"x\":1,\"x\":2}",
            "\"\\ud800\"",
            "true false",
            "1e1000",
            "\"\n\"",
        ] {
            assert!(Json::parse(s).is_err(), "{s}");
        }
    }
    #[test]
    fn json_depth_bound() {
        let s = "[".repeat(66) + "0" + &"]".repeat(66);
        assert!(Json::parse(&s).is_err());
    }
    #[test]
    fn sha256_vectors() {
        assert_eq!(
            digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }
    #[test]
    fn atomic_replace() {
        let t = Temp::new();
        let p = t.file("checkpoint", "old");
        atomic_write(&p, b"new").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"new");
        assert_eq!(fs::read_dir(&t.root).unwrap().count(), 1);
    }
    #[test]
    fn softmax_large_logits() {
        let p = softmax(&[10000., 9999., -10000.]);
        close(p.iter().sum(), 1., 1e-14);
        assert!(p.iter().all(|x| x.is_finite() && *x >= 0.));
    }
    #[test]
    fn rng_reproducible() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(1);
        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
    }
    #[test]
    fn reservoir_bounded() {
        let mut r = Reservoir::new(16, 42);
        for i in 0..10000 {
            r.push(i);
        }
        assert_eq!(r.items.len(), 16);
        assert_eq!(r.seen, 10000);
        assert!(r.items.iter().any(|i| *i > 16));
    }
    #[test]
    fn reservoir_zero_capacity() {
        let mut r = Reservoir::new(0, 1);
        r.push(1);
        assert!(r.items.is_empty());
    }
    #[test]
    fn tcp_axis_projection() {
        let t = transport_displacement(&[1., 2.], &[vec![1., 0.]], &[0., 0.], 2.).unwrap();
        close(t.delta[0], 0., 1e-12);
        close(t.delta[1], 2., 1e-12);
        assert!(!t.fallback);
    }
    #[test]
    fn tcp_feasible_identity() {
        let free = vec![-2., 3.];
        let r = transport_displacement(&free, &[vec![1., 0.]], &[10., 0.], 2.).unwrap();
        assert_eq!(r.delta, free);
    }
    #[test]
    fn tcp_opposing_constraints() {
        let rows = vec![vec![1., 0.], vec![-1., 0.], vec![0., 1.]];
        let r = transport_displacement(&[2., 3.], &rows, &[1., 4.], 2.).unwrap();
        assert!(rows.iter().all(|x| dot(x, &r.delta) < 1e-8));
    }
    #[test]
    fn tcp_random_constraints() {
        let mut rng = Rng::new(123);
        for _ in 0..20 {
            let free: Vec<_> = (0..16).map(|_| rng.normal()).collect();
            let rows: Vec<Vec<f64>> = (0..6)
                .map(|_| (0..16).map(|_| rng.normal()).collect())
                .collect();
            let result = transport_displacement(&free, &rows, &vec![1.; 16], 2.).unwrap();
            assert!(rows.iter().all(|r| dot(r, &result.delta) <= 1e-7 * norm(r)));
        }
    }
    #[test]
    fn tcp_invalid_dimensions() {
        assert!(transport_displacement(&[1.], &[vec![1., 2.]], &[1.], 2.).is_err());
        assert!(transport_displacement(&[f64::NAN], &[], &[1.], 2.).is_err());
    }
    #[test]
    fn constitutional_orthonormal() {
        for n in [1, 2, 3, 17] {
            let rows = constitutional_rows(n);
            for (i, a) in rows.iter().enumerate() {
                close(norm(a), 1., 1e-12);
                for b in rows.iter().skip(i + 1) {
                    close(dot(a, b), 0., 1e-12);
                }
            }
        }
    }
    #[test]
    fn dendritic_gradients_finite_difference() {
        let mut core = DendriticCore::new(3, 2, 2, 2, 44).unwrap();
        core.grow(0).unwrap();
        let u = vec![0.3, -0.5, 0.9];
        let (_, g) = core.gradients(&u, 1).unwrap();
        for name in core.p.keys().cloned().collect::<Vec<_>>() {
            let indices = [0, core.p[&name].len() / 2, core.p[&name].len() - 1];
            for i in indices {
                let original = core.p[&name][i];
                core.p.get_mut(&name).unwrap()[i] = original + 1e-6;
                let plus = core.gradients(&u, 1).unwrap().0;
                core.p.get_mut(&name).unwrap()[i] = original - 1e-6;
                let minus = core.gradients(&u, 1).unwrap().0;
                core.p.get_mut(&name).unwrap()[i] = original;
                close(g[&name][i], (plus - minus) / 2e-6, 2e-7);
            }
        }
    }
    #[test]
    fn dendritic_split_preserves_output() {
        let mut core = DendriticCore::new(2, 3, 3, 4, 1).unwrap();
        let before = core.forward(&[0.4, 0.7]).unwrap().output;
        core.grow(0).unwrap();
        let after = core.forward(&[0.4, 0.7]).unwrap().output;
        for (a, b) in before.iter().zip(after) {
            close(*a, b, 1e-14);
        }
        core.grow(0).unwrap();
        core.grow(0).unwrap();
        assert!(!core.grow(0).unwrap());
    }
    #[test]
    fn seq_gradient_every_group() {
        for budget in [0, 2] {
            let mut m = tiny();
            m.config.thought_budget = budget;
            let c: Vec<_> = (0..48).map(|i| i as f64 * 0.002).collect();
            let (_, g) = m.loss_grad("olá", &c, "oi").unwrap();
            for name in m.p.keys().cloned().collect::<Vec<_>>() {
                let len = m.p[&name].len();
                for i in [0, len / 3, len - 1] {
                    let original = m.p[&name][i];
                    m.p.get_mut(&name).unwrap()[i] = original + 1e-6;
                    let a = m.loss_grad("olá", &c, "oi").unwrap().0;
                    m.p.get_mut(&name).unwrap()[i] = original - 1e-6;
                    let b = m.loss_grad("olá", &c, "oi").unwrap().0;
                    m.p.get_mut(&name).unwrap()[i] = original;
                    close(g[&name][i], (a - b) / 2e-6, 2e-7);
                }
            }
        }
    }
    #[test]
    fn seq_encoder_embedding_gradient() {
        let mut m = tiny();
        let idx = m.lookup[&'o'] * m.dimension + 2;
        let (_, g) = m.loss_grad("oi", &zero_context(), "a").unwrap();
        let x = m.p["E"][idx];
        m.p.get_mut("E").unwrap()[idx] = x + 1e-6;
        let a = m.loss_grad("oi", &zero_context(), "a").unwrap().0;
        m.p.get_mut("E").unwrap()[idx] = x - 1e-6;
        let b = m.loss_grad("oi", &zero_context(), "a").unwrap().0;
        close(g["E"][idx], (a - b) / 2e-6, 1e-7);
    }
    #[test]
    fn seq_encoder_decoder_inspection() {
        let mut m = tiny();
        m.config.thought_budget = 3;
        let (_, j) = m.generate("oi", &zero_context()).unwrap();
        assert_eq!(
            j.get("latent_iterations").usize().unwrap(),
            j.get("tokens").usize().unwrap() * 3
        );
        for row in j.get("cross_attention").array().unwrap() {
            close(numeric_vec(row).unwrap().iter().sum(), 1., 1e-12);
        }
        assert_eq!(j.get("encoder_states").array().unwrap().len(), 2);
    }
    #[test]
    fn seq_inference_pure() {
        let m = tiny();
        let before = m.signature();
        let a = m.generate("teste", &zero_context()).unwrap().0;
        let b = m.generate("teste", &zero_context()).unwrap().0;
        assert_eq!(a, b);
        assert_eq!(m.signature(), before);
    }
    #[test]
    fn seq_checkpoint_roundtrip() {
        let m = tiny();
        let restored =
            NativeSeq2Seq::from_json(&Json::parse(&m.to_json().encode()).unwrap()).unwrap();
        assert_eq!(m.signature(), restored.signature());
        assert_eq!(
            m.generate("oi", &zero_context()).unwrap().0,
            restored.generate("oi", &zero_context()).unwrap().0
        );
    }
    #[test]
    fn seq_checkpoint_bad_schema() {
        let m = tiny();
        let mut j = m.to_json();
        if let Json::Obj(o) = &mut j {
            if let Some(Json::Obj(p)) = o.get_mut("parameters") {
                p.remove("A");
            }
        }
        assert!(NativeSeq2Seq::from_json(&j).is_err());
    }
    #[test]
    fn seq_budget_validation() {
        let mut m = tiny();
        m.config.output_tokens = 512;
        m.config.thought_budget = 256;
        assert!(m.generate("x", &zero_context()).is_err());
        assert!(NativeSeq2Seq::new(7, 1).is_err());
    }
    #[test]
    fn online_updates_real_weights() {
        let mut m = tiny();
        let before = m.signature();
        let r = train_pair(&mut m, &pair("ola", "oi"), &[], false, 0.1).unwrap();
        assert!(r.accepted);
        assert!(r.delta_l2 > 0.);
        assert!(r.loss_after < r.loss_before);
        assert_ne!(m.signature(), before);
    }
    #[test]
    fn online_constitution_actual_delta() {
        let mut m = tiny();
        let before = flatten(&m.p);
        let r = train_pair(&mut m, &pair("ola", "oi"), &[], true, 0.1).unwrap();
        let after = flatten(&m.p);
        let d: Vec<_> = after.iter().zip(before).map(|(a, b)| a - b).collect();
        for row in constitutional_rows(d.len()) {
            assert!(dot(&row, &d).abs() < 1e-7);
        }
        assert!(r.accepted);
    }
    #[test]
    fn online_replay_retention() {
        let mut m = tiny();
        let old = pair("a", "b");
        let loss = m.loss_grad("a", &zero_context(), "b").unwrap().0;
        let report = train_pair(&mut m, &pair("a", "z"), &[old], true, 0.1).unwrap();
        if report.accepted {
            assert!(m.loss_grad("a", &zero_context(), "b").unwrap().0 <= loss + 0.02);
        }
    }
    #[test]
    fn online_invalid_rate_no_mutation() {
        let mut m = tiny();
        let before = m.signature();
        assert!(train_pair(&mut m, &pair("a", "b"), &[], true, f64::NAN).is_err());
        assert_eq!(before, m.signature());
    }
    #[test]
    fn resized_schema() {
        let m = tiny();
        for n in [16, 8] {
            let resized = m.resized(n).unwrap();
            let restored = NativeSeq2Seq::from_json(&resized.to_json()).unwrap();
            assert_eq!(restored.dimension, n);
        }
    }
    #[test]
    fn proof_polynomial_identity() {
        let proof = symbolic_equivalence_check("(a+b)**2", "a**2+2*a*b+b**2").unwrap();
        assert_eq!(proof.get("proved"), &Json::Bool(true));
    }
    #[test]
    fn proof_rational_division() {
        assert_eq!(
            symbolic_equivalence_check("(x+1)/2", "x/2+1/2")
                .unwrap()
                .get("proved"),
            &Json::Bool(true)
        );
    }
    #[test]
    fn proof_negative_and_power_precedence() {
        assert_eq!(
            symbolic_equivalence_check("-x**2", "0-x*x")
                .unwrap()
                .get("proved"),
            &Json::Bool(true)
        );
        assert_eq!(
            PolyParser::parse("2**3**2").unwrap().scalar().unwrap().n,
            512
        );
    }
    #[test]
    fn proof_rejects_unsafe_or_unsupported() {
        for s in [
            "__import__('os')",
            "x/(x-1)",
            "x**10000",
            "1/0",
            "f(x)",
            "1.2",
            "[x]",
            "x.__class__",
        ] {
            assert!(PolyParser::parse(s).is_err(), "{s}");
        }
    }
    #[test]
    fn proof_overflow_is_error() {
        assert!(PolyParser::parse("170141183460469231731687303715884105727*2").is_err());
    }
    #[test]
    fn polynomial_derivative() {
        let p = PolyParser::parse("x**3+2*x*y+7")
            .unwrap()
            .derivative("x")
            .unwrap();
        assert_eq!(p, PolyParser::parse("3*x**2+2*y").unwrap());
    }
    #[test]
    fn prime_deterministic_u64() {
        for n in [
            2,
            3,
            97,
            2147483647,
            2305843009213693951,
            18446744073709551557,
        ] {
            assert!(is_prime(n), "{n}");
        }
        for n in [
            0,
            1,
            4,
            561,
            1105,
            3215031751,
            3825123056546413051,
            u64::MAX,
        ] {
            assert!(!is_prime(n), "{n}");
        }
    }
    #[test]
    fn factor_large_example() {
        let f = factor(600851475143).unwrap();
        assert_eq!(f, BTreeMap::from([(71, 1), (839, 1), (1471, 1), (6857, 1)]));
        assert_eq!(
            f.iter()
                .map(|(p, e)| (*p as u128).pow(*e))
                .product::<u128>(),
            600851475143
        );
    }
    #[test]
    fn factor_prime_and_zero() {
        assert_eq!(factor(97).unwrap(), BTreeMap::from([(97, 1)]));
        assert!(factor(0).is_err());
        assert!(factor(1).unwrap().is_empty());
    }
    #[test]
    fn exact_integer_json() {
        assert_eq!(uint(&js("18446744073709551615")).unwrap(), u64::MAX);
        assert!(uint(&jn(9007199254740992.)).is_err());
        assert_eq!(exact_int(18446744073709551615), js("18446744073709551615"));
    }
    #[test]
    fn number_theory_operations() {
        assert_eq!(calcular("nt.phi", &[jn(36.)]).unwrap(), jn(12.));
        assert_eq!(calcular("nt.mobius", &[jn(30.)]).unwrap(), jn(-1.));
        assert_eq!(calcular("nt.sigma", &[jn(12.)]).unwrap(), jn(28.));
        assert_eq!(calcular("nt.modinv", &[jn(3.), jn(11.)]).unwrap(), jn(4.));
        assert_eq!(
            calcular("nt.crt", &[ja(&[2., 3., 2.]), ja(&[3., 5., 7.])]).unwrap(),
            ja(&[23., 105.])
        );
    }
    #[test]
    fn combinatorics_operations() {
        assert_eq!(
            calcular("comb.binom", &[jn(10.), jn(3.)]).unwrap(),
            jn(120.)
        );
        assert_eq!(calcular("comb.catalan", &[jn(5.)]).unwrap(), jn(42.));
        assert_eq!(
            calcular("comb.stirling2", &[jn(5.), jn(2.)]).unwrap(),
            jn(15.)
        );
        assert_eq!(calcular("comb.bell", &[jn(5.)]).unwrap(), jn(52.));
        assert_eq!(calcular("comb.derangements", &[jn(5.)]).unwrap(), jn(44.));
    }
    #[test]
    fn solve_and_determinant() {
        let a = vec![vec![2., 1.], vec![1., 3.]];
        let x = solve(a.clone(), vec![5., 7.]).unwrap();
        close(x[0], 1.6, 1e-12);
        close(x[1], 1.8, 1e-12);
        close(determinant(a).unwrap(), 5., 1e-12);
        assert!(solve(vec![vec![1., 1.], vec![1., 1.]], vec![1., 2.]).is_err());
    }
    #[test]
    fn entropy_and_statistics() {
        close(
            calcular("it.entropy", &[ja(&[0.5, 0.5])])
                .unwrap()
                .number()
                .unwrap(),
            1.,
            1e-12,
        );
        assert!(calcular("it.entropy", &[ja(&[0.2, 0.2])]).is_err());
        assert_eq!(calcular("st.mean", &[ja(&[1., 2., 3.])]).unwrap(), jn(2.));
    }
    #[test]
    fn sat_with_witness_and_unsat() {
        let a = Json::Arr(vec![ja(&[1., 2.]), ja(&[-1., 2.]), ja(&[1., -2.])]);
        let r = calcular("lg.dpll", &[a]).unwrap();
        assert_eq!(r.get("satisfiable"), &Json::Bool(true));
        let b = Json::Arr(vec![ja(&[1.]), ja(&[-1.])]);
        assert_eq!(
            calcular("lg.dpll", &[b]).unwrap().get("satisfiable"),
            &Json::Bool(false)
        );
    }
    #[test]
    fn graph_negative_cycle() {
        let edges = Json::Arr(vec![ja(&[0., 1., 1.]), ja(&[1., 2., -2.])]);
        assert_eq!(
            calcular("gr.bellman_ford", &[jn(3.), edges, jn(0.)]).unwrap(),
            ja(&[0., 1., -1.])
        );
        let cycle = Json::Arr(vec![ja(&[0., 1., -1.]), ja(&[1., 0., -1.])]);
        assert!(calcular("gr.bellman_ford", &[jn(2.), cycle, jn(0.)]).is_err());
    }
    #[test]
    fn heat_cfl_and_boundaries() {
        let args = [ja(&[0., 1., 0.]), jn(1.), jn(0.5)];
        assert_eq!(calcular("de.heat1d", &args).unwrap(), ja(&[0., 0., 0.]));
        assert!(calcular("de.heat1d", &[ja(&[0., 1., 0.]), jn(1.), jn(0.6)]).is_err());
    }
    #[test]
    fn lorenz_finite() {
        let result = calcular("ds.lorenz", &[ja(&[1., 1., 1.]), jn(100.), jn(0.01)]).unwrap();
        assert!(numeric_vec(&result).unwrap().iter().all(|x| x.is_finite()));
    }
    #[test]
    fn missing_operation_explicit() {
        assert!(calcular("cat.yoneda_check", &[])
            .unwrap_err()
            .contains("não portada"));
    }
    #[test]
    fn dataset_json_and_alpaca() {
        let t = Temp::new();
        let p = t.file(
            "data.json",
            "[{\"instruction\":\"Diga\",\"input\":\"oi\",\"output\":\"Olá\"}]",
        );
        let data = import_dataset(&p, 10).unwrap();
        assert_eq!(data[0].input, "Diga\noi");
        assert_eq!(data[0].output, "Olá");
    }
    #[test]
    fn dataset_messages_and_ndjson() {
        let t = Temp::new();
        let p=t.file("data.ndjson","{\"messages\":[{\"role\":\"user\",\"content\":\"oi\"},{\"role\":\"assistant\",\"content\":\"olá\"}]}\n");
        let data = import_dataset(&p, 10).unwrap();
        assert_eq!(data[0].input, "oi");
        assert_eq!(data[0].output, "olá");
    }
    #[test]
    fn dataset_csv_quoted_newline() {
        let t = Temp::new();
        let p = t.file(
            "data.csv",
            "question,answer\r\n\"a,b\",\"x\ny\"\r\n\"z\",\"a\"\"b\"\n",
        );
        let d = import_dataset(&p, 10).unwrap();
        assert_eq!(d[0].input, "a,b");
        assert_eq!(d[0].output, "x\ny");
        assert_eq!(d[1].output, "a\"b");
    }
    #[test]
    fn dataset_tsv_and_text() {
        let t = Temp::new();
        let p = t.file("data.tsv", "input\toutput\nx\ty\n");
        assert_eq!(import_dataset(&p, 1).unwrap()[0].output, "y");
        let p = t.file("book.md", "Texto de treino.\n\nOutro texto.");
        assert_eq!(import_dataset(&p, 10).unwrap().len(), 2);
    }
    #[test]
    fn dataset_rejects_bad_and_binary() {
        let t = Temp::new();
        let p = t.file("data.parquet", "PAR1");
        assert!(import_dataset(&p, 1).unwrap_err().contains("não portado"));
        let p = t.file("data.csv", "x,x\na,b");
        assert!(import_dataset(&p, 1).is_err());
        let p = t.file("data.jsonl", "{\"input\":\"a\",\"output\":\"b\"}\nbad");
        assert!(import_dataset(&p, 1).unwrap_err().contains("Linha 2"));
    }
    #[test]
    fn dataset_reservoir_limit() {
        let t = Temp::new();
        let text = (0..100)
            .map(|i| obj([("input", js(format!("a{i}"))), ("output", js("b"))]).encode() + "\n")
            .collect::<String>();
        let p = t.file("d.jsonl", &text);
        assert_eq!(import_dataset(&p, 7).unwrap().len(), 7);
    }
    #[test]
    fn holdout_groups_disjoint() {
        let data = vec![
            pair("a", "x"),
            pair("a", "y"),
            pair("b", "z"),
            pair("c", "z"),
        ];
        let (train, test) = split_dataset(&data).unwrap();
        let prompts: BTreeSet<_> = train.iter().map(|p| &p.input).collect();
        assert!(test.iter().all(|p| !prompts.contains(&p.input)));
        assert!(split_dataset(&[pair("a", "b")]).is_err());
    }
    #[test]
    fn evolution_retains_original_on_rejection() {
        let mut m = tiny();
        let before = m.signature();
        let data = vec![pair("a", "b"), pair("c", "d")];
        let report = evolve_dataset(&mut m, &data, &[], true, 1, Some(16), 2.).unwrap();
        if !report.promoted {
            assert_eq!(m.signature(), before);
            assert_eq!(m.dimension, 8);
        } else {
            assert!(report.after < report.before);
            assert_eq!(m.dimension, 16);
        }
    }
    #[test]
    fn bpe_unicode_lossless() {
        let mut bpe = BytePairEncoding::new();
        let s = "olá olá olá 漢字 漢字 😀😀😀";
        let before = bpe.encode(s).len();
        bpe.train(s, 12).unwrap();
        let tokens = bpe.encode(s);
        assert!(tokens.len() < before);
        assert_eq!(bpe.decode(&tokens).unwrap(), s);
    }
    #[test]
    fn bpe_invalid_token() {
        assert!(BytePairEncoding::new().decode(&[256]).is_err());
    }
    #[test]
    fn hdc_deterministic_lexical() {
        let a = semantic_vector("gato preto", 256);
        let b = semantic_vector("GATO PRETO", 256);
        assert_eq!(a, b);
        close(norm(&a), 1., 1e-12);
        assert!(
            dot(&a, &semantic_vector("gato", 256)) > dot(&a, &semantic_vector("helicóptero", 256))
        );
    }
    #[test]
    fn vsa_bind_unbind_bipolar() {
        let a = vec![1., -1., 1., -1.];
        let b = vec![-1., 1., 1., -1.];
        let bound = vsa_bind(&a, &b).unwrap();
        assert_eq!(vsa_bind(&bound, &b).unwrap(), a);
    }
    #[test]
    fn history_roundtrip_tamper() {
        let t = Temp::new();
        let path = t.root.join("events.jsonl");
        let mut h = EventHistory::open(path.clone()).unwrap();
        h.append("test", js("a")).unwrap();
        h.append("test", js("b")).unwrap();
        let loaded = EventHistory::open(path.clone()).unwrap();
        assert_eq!(loaded.head, h.head);
        assert_eq!(loaded.sequence, 2);
        let text = fs::read_to_string(&path).unwrap().replace("\"a\"", "\"z\"");
        fs::write(&path, text).unwrap();
        assert!(EventHistory::open(path).is_err());
    }
    #[test]
    fn history_truncated_error() {
        let t = Temp::new();
        let path = t.file("events.jsonl", "{\"x\":1}");
        assert!(EventHistory::open(path).is_err());
    }
    #[test]
    fn profile_persists_weights_and_mode() {
        let t = Temp::new();
        let signature;
        {
            let mut p = Profile::open(&t.root, "alice").unwrap();
            p.restricted = false;
            let r = p.learn(&pair("ola", "oi")).unwrap();
            assert!(r.accepted);
            p.save().unwrap();
            signature = p.model.signature();
        }
        let p = Profile::open(&t.root, "alice").unwrap();
        assert_eq!(p.model.signature(), signature);
        assert!(!p.restricted);
        assert_eq!(p.replay.len(), 1);
    }
    #[test]
    fn profile_isolation_and_lease() {
        let t = Temp::new();
        let mut a = Profile::open(&t.root, "alice").unwrap();
        let b = Profile::open(&t.root, "bob").unwrap();
        let signature = b.model.signature();
        assert!(Profile::open(&t.root, "alice").is_err());
        a.learn(&pair("x", "y")).unwrap();
        assert_eq!(b.model.signature(), signature);
        assert_ne!(a.directory, b.directory);
    }
    #[test]
    fn profile_path_validation() {
        let t = Temp::new();
        for name in ["../escape", "a/b", "..", "", "á"] {
            assert!(safe_profile_path(&t.root, name).is_err());
        }
    }
    #[cfg(unix)]
    #[test]
    fn profile_symlink_rejected() {
        let t = Temp::new();
        let outside = Temp::new();
        std::os::unix::fs::symlink(&outside.root, t.root.join("alice")).unwrap();
        assert!(Profile::open(&t.root, "alice").is_err());
    }
    #[test]
    fn peer_output_does_not_train() {
        let t = Temp::new();
        let mut app = Profiles::new(t.root.clone(), "alice").unwrap();
        app.switch("bob").unwrap();
        let a = app.opened["alice"].model.signature();
        let b = app.opened["bob"].model.signature();
        app.dialogue("alice", "bob", "oi").unwrap();
        assert_eq!(a, app.opened["alice"].model.signature());
        assert_eq!(b, app.opened["bob"].model.signature());
    }
    #[test]
    fn kernel_hot_swap_with_proof() {
        let t = Temp::new();
        let mut p = Profile::open(&t.root, "alice").unwrap();
        assert!(p.hot_kernel("(x+1)**2").is_ok());
        let before = p.kernel.clone();
        assert!(p.hot_kernel("x+1").is_err());
        assert_eq!(before, p.kernel);
    }
    #[test]
    fn natural_math_factor_and_proof() {
        let (op, args) = natural_math("Pode fatorar o número 600851475143 pra mim?")
            .unwrap()
            .unwrap();
        assert_eq!(op, "nt.factor");
        assert_eq!(uint(&args[0]).unwrap(), 600851475143);
        let (op, args) = natural_math("prove que (a+b)**2 é equivalente a a**2+2*a*b+b**2")
            .unwrap()
            .unwrap();
        assert_eq!(
            calcular(&op, &args).unwrap().get("proved"),
            &Json::Bool(true)
        );
    }
    #[test]
    fn benchmark_no_training_no_solver() {
        let t = Temp::new();
        let p = t.file(
            "bench.jsonl",
            "{\"question\":\"quanto é 2+2\",\"answer\":\"4\"}\n",
        );
        let m = tiny();
        let before = m.signature();
        let report = local_benchmark(&m, &p, 1).unwrap();
        assert_eq!(
            report.get("protocol").get("solver_used"),
            &Json::Bool(false)
        );
        assert_eq!(m.signature(), before);
        assert_eq!(report.get("count"), &jn(1.));
    }
    #[test]
    fn benchmark_multiple_choice_loglikelihood() {
        let t = Temp::new();
        let p = t.file(
            "bench.jsonl",
            "{\"question\":\"A?\",\"choices\":[\"a\",\"bbbb\"],\"answer_index\":0}\n",
        );
        let m = tiny();
        let r = local_benchmark(&m, &p, 1).unwrap();
        assert_eq!(
            r.get("results").array().unwrap()[0]
                .get("conditional_log_likelihood")
                .array()
                .unwrap()
                .len(),
            2
        );
    }
    #[test]
    fn command_effort_persists() {
        let t = Temp::new();
        {
            let mut app = Profiles::new(t.root.clone(), "alice").unwrap();
            command(&mut app, "/esforco 8 3").unwrap();
            assert_eq!(app.profile().model.config.thought_budget, 8);
            assert!(command(&mut app, "/esforco 999").is_err());
        }
        let p = Profile::open(&t.root, "alice").unwrap();
        assert_eq!(p.model.config.thought_budget, 8);
    }
    #[test]
    fn command_no_data_execution() {
        let t = Temp::new();
        let mut app = Profiles::new(t.root.clone(), "alice").unwrap();
        let before = app.profile().model.signature();
        assert!(command(&mut app, "/exec rm -rf /").is_err());
        assert_eq!(app.profile().model.signature(), before);
    }
    #[test]
    fn coverage_is_honest() {
        assert_eq!(
            coverage().get("complete_python_translation"),
            &Json::Bool(false)
        );
        assert!(!coverage()
            .get("missing_from_python")
            .array()
            .unwrap()
            .is_empty());
    }
    #[test]
    fn secure_token_length() {
        let a = secure_token().unwrap();
        let b = secure_token().unwrap();
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
        assert!(constant_time_eq(a.as_bytes(), a.as_bytes()));
        assert!(!constant_time_eq(a.as_bytes(), b.as_bytes()));
    }
    #[test]
    fn supervised_demo_runs() {
        let r = supervised_demo(40).unwrap();
        assert_eq!(r.get("steps"), &jn(40.));
        assert!(r.get("mean_loss").number().unwrap().is_finite());
    }
    #[test]
    fn evolution_promotes_measured_gain() {
        let mut model = tiny();
        let pairs: Vec<_> = (0..12).map(|i| pair(&format!("a{i}"), "a")).collect();
        let report = evolve_dataset(&mut model, &pairs, &[], false, 2, None, 5.).unwrap();
        assert!(report.promoted);
        assert!(report.after < report.before);
        assert!(model.updates > 0);
    }
    #[test]
    fn profile_lock_can_reopen_after_drop() {
        let temp = Temp::new();
        {
            let _profile = Profile::open(&temp.root, "alice").unwrap();
        }
        assert!(Profile::open(&temp.root, "alice").is_ok());
    }
    fn http_exchange(request: impl FnOnce(u16) -> String) -> (String, Json) {
        use std::net::Shutdown;
        let temp = Temp::new();
        let app = Arc::new(Mutex::new(Profiles::new(temp.root.clone(), "web").unwrap()));
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = Arc::clone(&app);
        let task = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_http(stream, &server, "test-token", port).unwrap();
        });
        let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        client.write_all(request(port).as_bytes()).unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        task.join().unwrap();
        let state = app.lock().unwrap().state();
        (response, state)
    }
    #[test]
    fn http_requires_token_and_exact_host() {
        let (response, _) =
            http_exchange(|p| format!("GET /state HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\n\r\n"));
        assert!(response.starts_with("HTTP/1.1 403"));
        let (response, _) = http_exchange(|_| {
            "GET /state HTTP/1.1\r\nHost: attacker.example\r\nX-Neuron-Token: test-token\r\n\r\n"
                .into()
        });
        assert!(response.starts_with("HTTP/1.1 403"));
    }
    #[test]
    fn http_state_is_authenticated_and_read_only() {
        let (response, state) = http_exchange(|p| {
            format!(
                "GET /state HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nX-Neuron-Token: test-token\r\n\r\n"
            )
        });
        assert!(response.starts_with("HTTP/1.1 200"));
        assert_eq!(state.get("updates"), &js("0"));
        let body = response.split_once("\r\n\r\n").unwrap().1;
        assert_eq!(Json::parse(body).unwrap().get("profile"), &js("web"));
    }
    #[test]
    fn http_chat_math_and_saved_weights() {
        let body = obj([("prompt", js("calcule 2+2"))]).encode();
        let (response, state) = http_exchange(|p| {
            format!("POST /chat HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nX-Neuron-Token: test-token\r\nContent-Length: {}\r\n\r\n{body}",body.len())
        });
        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(response.contains("resultado exato é 4"));
        assert_ne!(state.get("updates"), &js("0"));
    }
    #[test]
    fn http_origin_and_duplicate_header_rejected() {
        let (response, _) = http_exchange(|p| {
            format!("POST /chat HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nOrigin: https://attacker.example\r\nX-Neuron-Token: test-token\r\nContent-Length: 0\r\n\r\n")
        });
        assert!(response.starts_with("HTTP/1.1 403"));
        let (response, _) = http_exchange(|p| {
            format!("GET /state HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nHost: duplicate\r\n\r\n")
        });
        assert!(response.starts_with("HTTP/1.1 400"));
    }
    #[test]
    fn checkpoint_detects_missing_history_prefix() {
        let temp = Temp::new();
        {
            let mut p = Profile::open(&temp.root, "alice").unwrap();
            p.save().unwrap();
        }
        fs::write(temp.root.join("alice/events.jsonl"), "").unwrap();
        assert!(Profile::open(&temp.root, "alice").is_err());
    }
    // Golden values computed by python/neuron.py SHA256 recorded in header.
    // This native regression test does not need Python or NumPy at test time.
    #[test]
    fn python_reference_seq2seq_golden() {
        let mut model = tiny();
        model.config.thought_budget = 2;
        for (group, (_, a)) in model.p.iter_mut().enumerate() {
            for (i, x) in a.iter_mut().enumerate() {
                *x = ((i as f64) * 0.17 + group as f64).sin() * 0.08;
            }
        }
        let context: Vec<_> = (0..48).map(|i| -0.1 + 0.2 * (i as f64) / 47.).collect();
        let (loss, g) = model.loss_grad("olá", &context, "oi").unwrap();
        close(loss, 4.839254453731153, 1e-12);
        close(norm(&g["A"]), 4.3934183685128715e-06, 1e-12);
        close(g["A"][31], -1.029696800092767e-06, 1e-12);
        close(norm(&g["B"]), 0.00010100968422492046, 1e-12);
        close(g["B"][0], -1.0918876117037079e-05, 1e-12);
        close(norm(&g["C"]), 1.070528520219504e-05, 1e-12);
        close(g["C"][7], 3.0174285260722664e-06, 1e-12);
        close(norm(&g["D"]), 0.0098935709056744, 1e-12);
        close(g["D"][41], 0.002047230565842061, 1e-12);
        close(norm(&g["E"]), 0.0004730213998113078, 1e-12);
        close(g["E"][0], 0.0001583955651358042, 1e-12);
        close(norm(&g["O"]), 0.0031921608324104594, 1e-12);
        close(g["O"][665], 0.0011252198779028622, 1e-12);
        close(norm(&g["b"]), 0.5703291112126273, 1e-12);
        close(g["b"][1], -0.32571520918295443, 1e-12);
        let (_, inspection) = model.generate("olá", &context).unwrap();
        let attention =
            numeric_vec(&inspection.get("cross_attention").array().unwrap()[0]).unwrap();
        close(attention[0], 0.3287564939199933, 1e-12);
        close(attention[1], 0.33425178739063865, 1e-12);
        close(attention[2], 0.336991718689368, 1e-12);
    }
    #[test]
    fn dendritic_log_responsibility_survives_underflow() {
        let mut core = DendriticCore::new(1, 1, 2, 2, 1).unwrap();
        core.grow(0).unwrap();
        for a in core.p.values_mut() {
            a.fill(0.);
        }
        core.p
            .get_mut("beta")
            .unwrap()
            .copy_from_slice(&[0., -1000.]);
        core.p
            .get_mut("d")
            .unwrap()
            .copy_from_slice(&[0., -2000., -2000., 0.]);
        let (loss, g) = core.gradients(&[0.], 1).unwrap();
        close(loss, 1000., 1e-10);
        assert!(g.values().flatten().all(|x| x.is_finite()));
        close(g["beta"][1], -1., 1e-12);
    }
}
