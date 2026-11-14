const fs=require('fs'); const crypto=require('crypto');
global.window={addEventListener(){}}; global.document={addEventListener(){}}; global.performance=performance;
global.addEventListener=()=>{}; global.navigator={}; global.innerWidth=1; global.innerHeight=1;
eval(fs.readFileSync('static/collector.js','utf8'));
let bad=0;
for (const s of ["", "abc", "nonce:123", "a".repeat(55), "a".repeat(56), "a".repeat(64), "x".repeat(200)]) {
  const w=window.__sha256(s); const hex=w.map(x=>(x>>>0).toString(16).padStart(8,'0')).join('');
  const ref=crypto.createHash('sha256').update(s).digest('hex'); if(hex!==ref){bad++;console.log('MISMATCH',s.length)}
}
console.log(bad?'JS sha256 FAILED':'JS sha256 matches node crypto on 7 inputs');
