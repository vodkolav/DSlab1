

window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.temperatureInCelsius = function(value) {
     return ((value - 32) * 5/9).toFixed(2);
}

window.dccFunctions.decScale = function(value) {
     return (10 ** value).toFixed(2);
}