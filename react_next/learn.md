Today I learned about props, they are basically function arguments.

```ts
function Header(props) {
  const title = props.title
  return <div>{title ? title : 'Default Title'}</div>
}

function HomePage() {
  const names = ['Ada Lovelace', 'Grace Hopper', 'Margaret Hamilton']

  return (
    <div>
      <Header title="Develop. Preview. Ship." />
      <ul>
        {names.map(name => (
          <li>{name}</li>
        ))}
      </ul>
    </div>
  )
}
```
In the above function we are basically passing the title value when calling Header tag, which is indeed a function
In React - every function can act as a HTML tag, you can do all your processing inside the function and generating dynamic 
HTML tags to render it on the website.