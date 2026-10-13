import { runAutomatedWarehouseTests } from './runTests';

console.log('===========================================================');
console.log('   AUTONOMOUS WAREHOUSE — TS ACCEPTANCE TEST SUITE (23)');
console.log('===========================================================');

const results = runAutomatedWarehouseTests();
let passedCount = 0;
let failedCount = 0;

results.forEach(res => {
  if (res.passed) {
    passedCount++;
    console.log(`[PASS] Test #${res.id.toString().padStart(2, '0')}: ${res.name}`);
  } else {
    failedCount++;
    console.error(`[FAIL] Test #${res.id.toString().padStart(2, '0')}: ${res.name} — ${res.message || 'Assertion failed'}`);
  }
});

console.log('-----------------------------------------------------------');
console.log(`SUMMARY: Total: ${results.length} | Passed: ${passedCount} | Failed: ${failedCount}`);
console.log('===========================================================');

if (failedCount > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
