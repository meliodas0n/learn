// why is it an address to the string ?
// or is it a different concept in rust ?
fn println(statement: &str) {
  println!("{}", statement);
}

fn main() {
  println("Hellow, World!");
}